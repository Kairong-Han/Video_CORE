Code and Dataset for our paper 
"Cross-modal Observation Retrodiction Reasoning".

## Running Code

python -m torch.distributed.launch --nproc_per_node=4 --epochs=50 --lr 1e-4 --batch_size=128 


## Dataset Link (Larger than 100GB)

https://pan.baidu.com/s/1SZ1EqEtlz1Wk5HuI7xDvoA?pwd=4kdw 


## Dataset Introduction

Retroductive reasoning aims at proposing the most probable hypotheses to elucidate incomplete observation, which encapsulates a central pillar of human cognitive abilities to understand our surroundings. However, the exploration of the multi-modal AI system for the retroductive reasoning remains in the early exploration stage. To facilitate the development in this AI field, we propose a novel task, Cross-modal Observation Retrodiction rEasoning (CORE). An example of our task is shown in the following figure. We propose the corresponding dataset, Video-CORE, for the CORE task. The dataset consists of about 10,000 carefully collected videos, with 39,796 retroductive reasoning samples meticulously annotated by annotators with strong logical abilities. Our Video-CORE dataset contains the following targeted designs: (1) Commonsense Knowledge Annotation. We have incorporated a diverse range of commonsense knowledge annotations pertaining to the video characters described by the textural observation. These annotations, covering aspects such as appearance, attire, actions, and emotional state, are aimed at enhancing the model's heterogeneous alignment of retrieved video and textural observation. (2) Action Flow Annotation. Each task example is annotated with a future event, and the sequence of actions that transpire post the video event and preceding the textual event. In addition, the Video-CORE dataset has the potential to advance broader task evaluations, such as temporally displaced text-to-video retrieval as elaborated in the appendix, thereby fostering deeper understanding and development in this area.

## File Organization Structure

- Videos
    - (All video files item by item.)

- CORE_Annotations
    - TrainFile_CORE.json
    - ValFile_CORE.json
    - TestFile_CORE.json

- Commonsense_Annotations
    - TrainFile_commonsense.json
    - ValFile_commonsense.json
    - TestFile_commonsense.json

## Annotation File Format

There are two types of the annotations included in our dataset, including the CORE task annotations and the the commonsense knowledge annotations. The file format of the annotations is shown in the following:

- The file format of the CORE task annotations:
    - The ***person numbering*** is determined based on the order of appearance of the characters. If they appear simultaneously, they are sorted from left to right based on the first frame they appear in.


```json
{
    "RGHT9":{  // video name
        "person 2":{  // person numbering, 
            "description 1":{  // example description numbering
                "The long hair woman in a gray and black hoodie is drying clothes."  // textual observation
                [
                    "wring out"  // action flow
                ],
                [
                    "wash", 
                    "clothes"    // video action
                ]
            }
        }
    }
}
```


- The file format of the commonsense knowledge annotations:

(Because the annotations are lengthy, we are not displaying them in the README but adding comments in the commonsense knowledge file, RGHT9_commonsense_annotation.json.)


## Statistics of Dataset Annotations

![The statistics of our Video-CORE dataset annotations](./statistics.pdf)
