# GIGAS: Adversarial Attacks on Visual Question Answering with Multi-modal Generative Models

This repository provides the implementation for testing GIGAS, a framework designed for adversarial attacks, through the Visual Question Answering (VQA) task on the VQAv2 dataset. Detailed operational guidelines are outlined as follows:


## Pre-trained Model Preparation

First, download the pre-trained model weights for the BLIP framework—specifically the BLIP variant equipped with a ViT-B backbone (14 million parameters)—from the [official BLIP repository](https://github.com/salesforce/BLIP). After downloading, modify the `pretrained` field in `./configs/pretrain.yaml`. These pre-trained weights serve as a surrogate model for generating adversarial samples in our research.

## Related Model Preparation

1. LLaVA:

   ```
   hf download llava-hf/llava-v1.6-mistral-7b-hf
   ```

2. Stable Diffusion:

   ```
   hf download stabilityai/stable-diffusion-xl-refiner-1.0
   ```

3. DPT:

   ```
   git clone https://github.com/LYuhang/DPT.git
   ```

## Attack Execution on the VQAv2 Dataset

1. Download the VQAv2 dataset from its [official download page](https://visualqa.org/download.html). Once downloaded, navigate to the configuration file `./configs/vqa.yaml` and update the `vqa_root` to point to the local directory where the VQAv2 dataset is stored.  

2. Download the fine-tuned model weights tailored for the VQAv2 task from the [original BLIP repository](https://github.com/salesforce/BLIP). Specifically, the fine-tuned VQAv2 model weights can be downloaded directly from [this link](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_vqa.pth). After downloading, modify the `pretrain` field in `./configs/vqa.yaml` to specify the file path of the downloaded `model_vqa.pth` weight file.  

3. Identify samples that are correctly predicted by the base model by executing the script `python prepare_vqa.py`. Running this command will generate two output files: `right_vqa_list.txt` (which stores the indices of correctly predicted samples) and `right_vqa_ans_table.txt` (which records the corresponding prediction results for these samples).  

4. To generate adversarial images (Stage 1) on the VQAv2 dataset, run the `python attack_vqa.py` script and specify the attack strategy using the `--method` flag. The command is as follows:

  ```bash
  python attack_vqa.py --method prepare_image --correct_idx_store BLIP_VQA/right_vqa_list.txt --correct_pred_store BLIP_VQA/right_vqa_ans_table.txt --exp_file BLIP_VQA/result
  ```

5. To generate adversarial texts (Stage 2) on the VQAv2 dataset, run the `python attack_vqa.py` script and specify the attack strategy using the `--method` flag. The command is as follows:

  ```bash
  python attack_vqa.py --method prepare_text --correct_idx_store BLIP_VQA/right_vqa_list.txt --correct_pred_store BLIP_VQA/right_vqa_ans_table.txt --exp_file BLIP_VQA/result
  ```

6. To perform the full GIGAS-based adversarial attack (Stage 3) on the VQAv2 dataset, run the `python attack_vqa.py` script and specify the attack strategy using the `--method` flag. The command is as follows:

  ```bash
  python attack_vqa.py --method gigas_attack --correct_idx_store BLIP_VQA/right_vqa_list.txt --correct_pred_store BLIP_VQA/right_vqa_ans_table.txt --exp_file BLIP_VQA/result
  ```

### some codes are coming soon.
