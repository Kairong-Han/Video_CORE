from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
from torch import nn

from modules.until_module import PreTrainedModel, AllGather, CrossEn
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from modules.module_clip import CLIP, convert_weights
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import math
import random
import copy
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)
allgather = AllGather.apply

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.Linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.Linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.Linear3 = nn.Linear(hidden_dim, output_dim)
        # self.initialize()

    def forward(self, x):
        x = self.Linear3(x)
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

def cross_entropy(y_pred, y_true,t_v = None):
    y_pred = F.softmax(y_pred, dim=1)
    y_true = F.one_hot(y_true, num_classes=y_pred.shape[1]).float()
    loss = F.binary_cross_entropy(y_pred, y_true)

    return loss

class LossCalculate(nn.Module):
    def __init__(self ):
        super().__init__()
        self.cost = nn.CrossEntropyLoss(reduction = "sum")
        self.mse = nn.MSELoss()
        # self.score_label = torch.zeros()

    def forward(self, x , ylabel, t_v):

        frame_tensor_, scene_tensor_, attr_label1_, attr_label2_, attr_label3_, action_tensor_ = ylabel
        frame_tensor, scene_tensor, attr_label1, attr_label2, attr_label3, action_tensor, score = x
        valid_mask = t_v.view(-1) != 1
        count = torch.sum(t_v.view(-1)).item()
        print("count:",count)
        print(valid_mask.shape,valid_mask)
        loss1 = 0
        loss2 = 0
        loss3 = 0
        loss4 = 0
        loss5 = 0
        loss6 = 0
        if not torch.all(~valid_mask):
            loss1 = self.cost(frame_tensor[valid_mask].permute(0,2,1),frame_tensor_[valid_mask])
            loss2 = self.cost(scene_tensor[valid_mask].permute(0,2,1),scene_tensor_[valid_mask])
            loss3 = self.cost(attr_label1[valid_mask].permute(0,2,1),attr_label1_[valid_mask])
            loss4 = self.cost(attr_label2[valid_mask].permute(0,2,1),attr_label2_[valid_mask])
            loss5 = self.cost(attr_label3[valid_mask].permute(0,2,1),attr_label3_[valid_mask])
            loss6 = self.cost(action_tensor[valid_mask].permute(0,2,1),action_tensor_[valid_mask])
        print("cost shape", torch.cat((t_v.view(-1,1), score.view(-1,1)), dim=1))
        print("action cost:","\n理论输出",action_tensor_[valid_mask].view(-1),
              "\n实际输出",torch.argmax(action_tensor[valid_mask], dim=2).view(-1),
              "\n理论概率",torch.gather(action_tensor[valid_mask], dim=2, index=action_tensor_[valid_mask].unsqueeze(2)).view(-1),
              "\n实际概率",torch.gather(action_tensor[valid_mask], dim=2, index=torch.argmax(action_tensor[valid_mask], dim=2).unsqueeze(2)).view(-1))
        loss7 = self.mse(score.view(-1,1), t_v.float())
        print("loss", loss7.item())

        if count==0:
            print("loss6", 0)
        else:
            print("loss6", loss6.item())
        if count == 0:
            return loss7
        else:
            loss_sum = 0.0001*(loss1+loss2+loss3+loss4+loss5)
            print("other loss",loss_sum.item())
            print("sum loss",(loss6 + loss7 +loss_sum).item())
            return loss6 + loss7

class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        pretrained_clip_name = "ViT-B/32"
        if hasattr(task_config, 'pretrained_clip_name'):
            pretrained_clip_name = task_config.pretrained_clip_name
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

        ## ===> Initialization trick [HARD CODE]
        if model.linear_patch == "3d":
            contain_conv2 = False
            for key in state_dict.keys():
                if key.find("visual.conv2.weight") > -1:
                    contain_conv2 = True
                    break
            if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                cp_weight = state_dict["clip.visual.conv1.weight"].clone()
                kernel_size = model.clip.visual.conv2.weight.size(2)
                conv2_size = model.clip.visual.conv2.weight.size()
                conv2_size = list(conv2_size)

                left_conv2_size = conv2_size.copy()
                right_conv2_size = conv2_size.copy()
                left_conv2_size[2] = (kernel_size - 1) // 2
                right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

                left_zeros, right_zeros = None, None
                if left_conv2_size[2] > 0:
                    left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
                if right_conv2_size[2] > 0:
                    right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                cat_list = []
                if left_zeros != None: cat_list.append(left_zeros)
                cat_list.append(cp_weight.unsqueeze(2))
                if right_zeros != None: cat_list.append(right_zeros)
                cp_weight = torch.cat(cat_list, dim=2)

                state_dict["clip.visual.conv2.weight"] = cp_weight

        if model.sim_header == 'tightTransf':
            contain_cross = False
            for key in state_dict.keys():
                if key.find("cross.transformer") > -1:
                    contain_cross = True
                    break
            if contain_cross is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                        continue
                    if key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict["cross."+key] = val.clone()
                            continue

        if model.sim_header == "seqLSTM" or model.sim_header == "seqTransf":
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            key_copy  = key[:]
                            state_dict[key_copy.replace("transformer.","transformerClip2.")] = val.clone()
                            state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
        ## <=== End of initialization trick

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]


def Dijkstra(text_action, action, MAX_value=10):
    if action is None: return None
    start_node = text_action
    S = []
    Q = [i for i in range(len(action))]
    dist = [MAX_value] * len(action)
    dist[start_node] = 0
    dist_init = [i for i in action[start_node]]
    dist_path = [-1] * len(action)
    while Q:
        u_dist = {v: d for v, d in enumerate(dist_init) if v in Q}
        u = min(u_dist, key=u_dist.get)
        S.append(u)
        Q.remove(u)
        for v, d in enumerate(action[u]):
            if 0 < d < MAX_value:
                if dist[v] > dist[u] + d:
                    dist[v] = dist[u] + d
                    dist_init[v] = dist[v]
                    dist_path[v] = u
    distance = []
    for i in range(len(action)):
        if (dist[i] != 0 and dist[i] != 10): distance.append(i)
    action_path = [[] for i in range(len(dist_path))]

    for i in range(len(dist_path)):
        cur_num = i
        if (dist_path[cur_num] != -1):
            while dist_path[cur_num] != -1:
                action_path[i].append(cur_num)
                cur_num = dist_path[cur_num]
            action_path[i].append(start_node)
    distance_new = copy.deepcopy(distance)
    
    for i in range(len(action_path)):
        if len(action_path[i]) != 0:
            for j in range(len(action_path[i])):
                if j != 0 and j != len(action_path[i]) - 1:
                    if action_path[i][j] in distance_new:
                        distance_new.remove(action_path[i][j])
    action_path_new = [[] for i in range(len(dist_path))]
    for i in range(len(action_path)):
        if i in distance_new:
            action_path_new[i] = action_path[i]
    return distance, action_path

def get_maximum_rate(ret, text_action_list, video_action_list):
    query_ret = []
    if (len(ret) == 0):
        query_ret.append(video_action_list.index("none"))
    else:
        for k in range(len(ret)):
            dis1 = text_action_list[ret[k]]

            if (dis1 == "walk out"):
                query_ret.append(video_action_list.index("walk out from"))
                continue
            if (dis1 == "pour into"):
                query_ret.append(video_action_list.index("pour"))
                continue
            if (dis1 == "answer"):
                query_ret.append(video_action_list.index("answer phone"))
                continue
            if (dis1 == "put sth into sw" or dis1 == "put on" or dis1 == "put"):
                query_ret.append(video_action_list.index("put sth into/on sw"))
                continue
            if (dis1 == "washing"):
                query_ret.append(video_action_list.index("wash"))
                continue
            if (dis1 == "take sth from sw"):
                query_ret.append(video_action_list.index("take sth from sb/sw"))
                continue
            if (dis1 == "wring out" or dis1 == "go to hospital" or dis1 == "dry" or dis1 == "filp" or dis1 == "filp" or dis1 == "spit out"
                    or dis1 == "turn back" or dis1 == "cut" or dis1 == "laugh" or dis1 == "turn on" or dis1 == "read" or dis1 == "have a meal"):
                query_ret.append(video_action_list.index("none"))
                continue
            if(dis1 in video_action_list):
                query_ret.append(video_action_list.index(dis1))
            else: print(dis1)
    return query_ret

def return_path(video_action, ret_path, video_action_list, text_action_list):
    dis1 = video_action_list[video_action]
    if (dis1 == "walk out from"):        dis2 = "walk out"
    elif (dis1 == "pour"):                 dis2 = "pour into"
    elif (dis1 == "answer phone"):         dis2 = "answer"
    elif (dis1 == "put sth into/on sw"):
        if len(ret_path[text_action_list.index("put")]) != 0: dis2 = "put"
        if len(ret_path[text_action_list.index("put on")]) != 0: dis2 = "put on"
        if len(ret_path[text_action_list.index("put sth into sw")]) != 0: dis2 = "put sth into sw"
    elif (dis1 == "wash"):
        if len(ret_path[text_action_list.index("wash")]) != 0: dis2 = "wash"
        if len(ret_path[text_action_list.index("washing")]) != 0: dis2 = "washing"
    elif (dis1 == "take sth from sb/sw"):  dis2 = "take sth from sw"

    elif (dis1 == "run" or dis1 == "yawn" or dis1 == "jump" or dis1 == "hit" or dis1 == "grab" or dis1 == "make up"
            or dis1 == "cover" or dis1 == "watch" or dis1 == "tie" or dis1 == "untie" or dis1 == "play"):
        dis2 = "none"
    else: dis2 = dis1
    action_path = ret_path[text_action_list.index(dis2)]
    if(len(action_path) == 0): action_path.append(text_action_list.index("none"))
    return action_path


def Mask_Chain(action_path):
    assert len(action_path)>=0 and len(action_path)<=10
    mask = torch.zeros(10+1)
    mask[:len(action_path)+1] = 1
    action_chain = torch.zeros(10)
    action_chain[:len(action_path)] = torch.tensor(action_path)
    return action_chain,mask

class CLIP4Clip(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(CLIP4Clip, self).__init__(cross_config)
        """model = cls(cross_config, clip_state_dict, *inputs, **kwargs)"""
        self.task_config = task_config
        self.ignore_video_index = -1
        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        self.d_model = transformer_width
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))
        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            linear_patch=self.linear_patch
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        self.sim_header = 'meanP'
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        if self.sim_header == "tightTransf": assert self.loose_type is False

        cross_config.max_position_embeddings = context_length
        if self.loose_type is False:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header == "seqLSTM" or self.sim_header == "seqTransf":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
        if self.sim_header == "seqTransf":
            self.transformerClip = TransformerClip(width=transformer_width, layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )
        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                       batch_first=True, bidirectional=False, num_layers=1)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_width, nhead=transformer_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        query_vector_dim = vision_layers + 3

        self.embed = nn.Embedding(15,512)
        self.query_vector = self.embed.weight

        self.embed_actionchain = nn.Embedding(1,512)
        self.chain_query = self.embed_actionchain.weight
        self.mlp_frame_emotion = MLP(transformer_width,transformer_width,8)
        self.mlp_frame_scene = MLP(transformer_width,transformer_width,22)
        self.mlp_video_up_clothing = MLP(transformer_width,transformer_width,8)

    def myposition_encoder(self,x,slen,dmodel):
        position = torch.arange(0, slen).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dmodel, 2).float() * (-math.log(10000.0) / dmodel))
        pe = torch.zeros(slen, dmodel,device = x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).expand(x.shape[0],pe.shape[0],pe.shape[1])
        pe = nn.Parameter(pe)
        return nn.Parameter(x+pe)

    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None,label = None,t_v = None,text_label = None,action = None,action_list = None,video_action_list = None,text_action_list = None):

        _, _, _, _, _, action_tensor_label = label

        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts
        print(attention_mask.shape,video_mask.shape)
        sequence_output, visual_output = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask,
                                                                         video, video_mask, shaped=True, video_frame=video_frame)
        ret_list = []
        ret_path_list = []
        query_ret_list = []

        for idx in range(0, text_label.shape[0]) :
            ret, ret_path = Dijkstra(text_label[idx,0], action)
            query_ret = get_maximum_rate(ret, text_action_list, video_action_list)
            ret_list.append(ret)
            ret_path_list.append(ret_path)
            query_ret_list.append(query_ret)

        transformer_input = torch.cat((sequence_output,visual_output,self.query_vector.unsqueeze(0).expand(sequence_output.shape[0],self.query_vector.shape[0],self.query_vector.shape[1])),dim=1)
        transformer_input = self.myposition_encoder(transformer_input,28,512)


        transformer_input = transformer_input.permute(1, 0, 2)
        transformer_mask = torch.cat((torch.ones(attention_mask.shape[0],
                                                 1,
                                                 device = attention_mask.device)
                                      ,video_mask,
                                      torch.ones(attention_mask.shape[0],
                                                self.query_vector.shape[0],
                                                device = attention_mask.device))
                                     ,dim = 1)
        mask2 = torch.zeros(attention_mask.shape[0],28,28,device = attention_mask.device)
        transformer_input = self.transformerClip(transformer_input,mask2)

        transformer_input = transformer_input.permute(1,0,2)
        print("Transformer output")

        frame_start = visual_output.shape[1] + sequence_output.shape[1]
        frame_end = sequence_output.shape[1] + visual_output.shape[1] + visual_output.shape[1]
        output_frame_scene = self.mlp_frame_scene(transformer_input[:,frame_start:frame_end,:])
        output_frame_emotion = self.mlp_frame_emotion(transformer_input[:,frame_start:frame_end,:])
        output_video_age_sex = self.mlp_video_age_sex(transformer_input[:,frame_end:frame_end+1,:])
        output_video_up_clothing = self.mlp_video_up_clothing(transformer_input[:,frame_end:frame_end+1,:])
        output_video_down_clothing = self.mlp_video_down_clothing(transformer_input[:,frame_end:frame_end+1,:])
        output_video_action = self.mlp_video_action(transformer_input[:,frame_end+1:frame_end+2,:])

        mean = [0.5]
        std = [0.5]
        normalize = transforms.Normalize(mean=mean, std=std)
        output_video_action = normalize(output_video_action)
        output_video_action = self.softmax(output_video_action)

        id_list = []
        for item in range(0, len(query_ret_list)):
            
            max_idx = -1
            max_val = 0
            for action_select in query_ret_list[item]:
                id_temp = action_select
                print("before selected",output_video_action[item, 0, :].view(-1))
                if output_video_action[item, 0, id_temp] > max_val:

                    max_val = output_video_action[item,0,id_temp]
                    max_idx = id_temp

            print("max_idx",max_idx,"item",query_ret_list[item],"text label",text_label[item,0],"action label",action_tensor_label[item,0],"\n")
            assert max_idx != -1
            id_list.append(max_idx)
        assert len(id_list) == len(query_ret_list) == output_video_action.shape[0]

        print("id list over")
        action_path_tensor = None
        mask_chain = None

        for idx in range(0,len(id_list)):
            action_path = return_path(id_list[idx], ret_path_list[idx],video_action_list,text_action_list)
            action_path,mask_temp = Mask_Chain(action_path)
            if idx == 0:
                action_path_tensor = action_path.unsqueeze(0)#1*M
                mask_chain = mask_temp.unsqueeze(0)
            else:
                action_path_tensor = torch.cat((action_path_tensor,action_path.unsqueeze(0)),dim=0)#N*M
                mask_chain = torch.cat((mask_chain,mask_temp.unsqueeze(0)),dim = 0)
        print("action_path",action_path_tensor.shape,"mask",mask_chain.shape)

        action_path_tensor = action_path_tensor.to(attention_mask.device)
        action_path_tensor =  self.action_embed(action_path_tensor.long())

        extended_video_mask = (1.0 - mask_chain.unsqueeze(1)) * -1000000.0
        extended_video_mask = extended_video_mask.expand(-1, mask_chain.size(1), -1)
        print("maskshape:",extended_video_mask.shape)
        compare_input = torch.cat((self.chain_query.unsqueeze(0).expand(action_path_tensor.shape[0],self.chain_query.shape[0],self.chain_query.shape[1]),action_path_tensor),dim = 1)
        compare_input = self.myposition_encoder(compare_input,compare_input.shape[1],512)
        print("compare_input",compare_input.shape)
        compare_input = compare_input.to(attention_mask.device)
        extended_video_mask = extended_video_mask.to(attention_mask.device)
        compare_input = compare_input.permute(1,0,2)
        compare_input = self.transformerClip2(compare_input, extended_video_mask)
        compare_input = compare_input.permute(1,0,2)
        # N*1*512->N*1*5
        chain_compress = self.mpl_chain_compress(compare_input[:,0:1,:])
        transformer_input[:, frame_end + 2:frame_end + 3, -5:] += chain_compress


        score_result = self.mlp_score(transformer_input[:, frame_end + 2:frame_end + 3, :])
        score_result = nn.functional.sigmoid(score_result)
        print(score_result)

        if self.training:
            loss_ = 0.
            loss_ = self.loss_cal((output_frame_scene,
                               output_frame_emotion,
                               output_video_age_sex,
                               output_video_up_clothing,
                               output_video_down_clothing,
                               output_video_action,
                               score_result),
                              label,t_v)
            return loss_
        else:
            return None

    def get_text_to_video_percent(self, input_ids, token_type_ids, attention_mask, video, video_mask=None,text_label = None,action = None,action_list = None,video_action_list = None,text_action_list = None):
    

        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts
        sequence_output, visual_output = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask,
                                                                         video, video_mask, shaped=True,
                                                                         video_frame=video_frame)
        ret_list = []
        ret_path_list = []
        query_ret_list = []
        text_action = []
        for idx in range(0, text_label.shape[0]):
            ret, ret_path = Dijkstra(text_label[idx, 0], action)
            query_ret = get_maximum_rate(ret, text_action_list, video_action_list)
            ret_list.append(ret)
            ret_path_list.append(ret_path)
            query_ret_list.append(query_ret)
        transformer_input = torch.cat((sequence_output, visual_output,
                                       self.query_vector.unsqueeze(0).expand(sequence_output.shape[0],
                                                                             self.query_vector.shape[0],
                                                                             self.query_vector.shape[1])), dim=1)
        transformer_input = self.myposition_encoder(transformer_input, 28, 512)
        transformer_input = transformer_input.permute(1, 0, 2)
        mask2 = torch.zeros(attention_mask.shape[0], 28, 28, device=attention_mask.device)
        transformer_input = self.transformerClip(transformer_input, mask2)
        transformer_input = transformer_input.permute(1, 0, 2)
        frame_start = visual_output.shape[1] + sequence_output.shape[1]
        frame_end = sequence_output.shape[1] + visual_output.shape[1] + visual_output.shape[1]
        output_video_action = self.mlp_video_action(transformer_input[:, frame_end + 1:frame_end + 2, :])
        mean = [0.5]
        std = [0.5]
        normalize = transforms.Normalize(mean=mean, std=std)
        output_video_action = normalize(output_video_action)
        output_video_action = self.softmax(output_video_action)
        id_list = []
        for item in range(0, len(query_ret_list)):
            max_idx = -1
            max_val = 0
            for action_select in query_ret_list[item]:
                id_temp = action_select
                if output_video_action[item, 0, id_temp] > max_val:
                    max_val = output_video_action[item, 0, id_temp]
                    max_idx = id_temp
            assert max_idx != -1
            id_list.append(max_idx)
        assert len(id_list) == len(query_ret_list) == output_video_action.shape[0]
        action_path_tensor = None
        mask_chain = None
        for idx in range(0, len(id_list)):
            action_path = return_path(id_list[idx], ret_path_list[idx], video_action_list, text_action_list)
            action_path, mask_temp = Mask_Chain(action_path)
            if idx == 0:
                action_path_tensor = action_path.unsqueeze(0)  # 1*M
                mask_chain = mask_temp.unsqueeze(0)
            else:
                action_path_tensor = torch.cat((action_path_tensor, action_path.unsqueeze(0)), dim=0)  # N*M
                mask_chain = torch.cat((mask_chain, mask_temp.unsqueeze(0)), dim=0)
        action_path_tensor = action_path_tensor.to(attention_mask.device)
        output_chain = action_path_tensor
        action_path_tensor = self.action_embed(action_path_tensor.long())
        extended_video_mask = (1.0 - mask_chain.unsqueeze(1)) * -1000000.0
        extended_video_mask = extended_video_mask.expand(-1, mask_chain.size(1), -1)
        compare_input = torch.cat((self.chain_query.unsqueeze(0).expand(action_path_tensor.shape[0],
                                                                        self.chain_query.shape[0],
                                                                        self.chain_query.shape[1]), action_path_tensor),
                                  dim=1)
        compare_input = self.myposition_encoder(compare_input, compare_input.shape[1], 512)
        compare_input = compare_input.to(attention_mask.device)
        extended_video_mask = extended_video_mask.to(attention_mask.device)
        compare_input = compare_input.permute(1, 0, 2)
        compare_input = self.transformerClip2(compare_input, extended_video_mask)
        compare_input = compare_input.permute(1, 0, 2)
        chain_compress = self.mpl_chain_compress(compare_input[:, 0:1, :])
        transformer_input[:, frame_end + 2:frame_end + 3, -5:] += chain_compress
        score_result = self.mlp_score(transformer_input[:, frame_end + 2:frame_end + 3, :])
        score_result = nn.functional.sigmoid(score_result)
        return score_result[:,0,0],output_chain,mask_chain

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_hidden = self.clip.encode_text(input_ids).float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))

        return sequence_hidden

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        visual_hidden = self.clip.encode_image(video, video_frame=video_frame).float()
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))

        return visual_hidden

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts
        sequence_output = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)
        visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)

        return sequence_output, visual_output

    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):

        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, pooled_output, concat_mask
    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        return text_out, video_out

    def _loose_similarity(self, sequence_output, visual_output, attention_mask, video_mask, sim_header="meanP"):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        if sim_header == "meanP":
            # Default: Parameter-free type
            pass
        elif sim_header == "seqLSTM":
            # Sequential type: LSTM
            visual_output_original = visual_output
            visual_output = pack_padded_sequence(visual_output, torch.sum(video_mask, dim=-1).cpu(),
                                                 batch_first=True, enforce_sorted=False)
            visual_output, _ = self.lstm_visual(visual_output)
            if self.training: self.lstm_visual.flatten_parameters()
            visual_output, _ = pad_packed_sequence(visual_output, batch_first=True)
            visual_output = torch.cat((visual_output, visual_output_original[:, visual_output.size(1):, ...].contiguous()), dim=1)
            visual_output = visual_output + visual_output_original
        elif sim_header == "seqTransf":
            # Sequential type: Transformer Encoder
            visual_output_original = visual_output
            seq_length = visual_output.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
            position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            visual_output = visual_output + frame_position_embeddings
            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
            visual_output = self.transformerClip(visual_output, extended_video_mask)
            visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
            visual_output = visual_output + visual_output_original

        if self.training:
            visual_output = allgather(visual_output, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config)
            torch.distributed.barrier()

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        logit_scale = self.clip.logit_scale.exp()
        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())
        return retrieve_logits

    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []

        step_size = b_text      # set smaller to reduce memory cost
        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        # due to clip text branch retrun the last hidden
        attention_mask = torch.ones(sequence_output.size(0), 1)\
            .to(device=attention_mask.device, dtype=attention_mask.dtype)

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_visual, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_visual, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)

            cross_output, pooled_output, concat_mask = \
                self._get_cross_output(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_visual)

            retrieve_logits_list.append(retrieve_logits_row)

        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits

    def get_similarity_logits(self, sequence_output, visual_output, attention_mask, video_mask, shaped=False, loose_type=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        contrastive_direction = ()
        if loose_type:
            assert self.sim_header in ["meanP", "seqLSTM", "seqTransf"]
            retrieve_logits = self._loose_similarity(sequence_output, visual_output, attention_mask, video_mask, sim_header=self.sim_header)
        else:
            assert self.sim_header in ["tightTransf"]
            retrieve_logits = self._cross_similarity(sequence_output, visual_output, attention_mask, video_mask, )

        return retrieve_logits, contrastive_direction
