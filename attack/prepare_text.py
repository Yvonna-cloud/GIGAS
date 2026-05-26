import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sys
from transformers import BertForMaskedLM, BertTokenizer
from transformers.models.bert.modeling_bert import BertConfig#, BertEmbeddings
bert_path = 'MODEL/bert-base-uncased'
config_atk = BertConfig.from_pretrained(bert_path)
import copy
import torch
import torch.nn
import utils
import torch.optim
from filter_words import filter_words
import nltk
from nltk.corpus import stopwords
from models.blip_itm import blip_itm
therehold_for_aisim = 0.95
ablation_temperature = 0.7

nltk.download('stopwords')
from nltk.corpus import stopwords
filter_words = filter_words + stopwords.words('english')+['?','.']
class Feature(object):
    def __init__(self, seq_a):
        # self.label = label
        self.seq = seq_a
        self.final_adverse = seq_a
        self.query = 0
        self.change = 0
        self.success = 0
        self.sim = 0.0
        self.changes = []
import json
import cleverhans.torch.attacks.BLIP.projected_gradient_descent as pgd
from torchvision.transforms import Resize
from torchvision import transforms

# from diffusers import DiffusionPipeline
import torch


# 定义转换操作：调整大小和转为张量
from torchvision.transforms.functional import InterpolationMode
normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))
transform = transforms.Compose([
    transforms.Resize((480,480),interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    normalize,
    ])  
from PIL import Image

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image

import torch.nn.functional as F
from scipy.spatial.distance import cosine

import os
class Adv_attack:
    def __init__(self, vqa_model,pretrain_model,tokenizer,device,correct_idx_list,correct_pred_list,USE_model,itm_model,exp_file):
        self.attack_dict = {}
        self.acc_list=[]
        self.tokenizer = tokenizer
        self.single_stg_step=20
        self.total_stg_step=40
        self.tokenizer_mlm = BertTokenizer.from_pretrained(bert_path,
                                                           do_lower_case="uncased" in bert_path)
        self.text_bank={}
        self.adv_txt_dict = {}
        self.cos_sim=0.95
        self.k=10
        self.text_budget = 100000
        self.correct_list = correct_idx_list
        self.blip_ans_table = correct_pred_list
        self.white_model=pretrain_model
        self.black_model=vqa_model
        self.USE_model=USE_model
        self.device=device
        self.batch=None
        self.captions=None
        self.vqa_score=0
        self.acc_list=[]
        self.mlm_model = BertForMaskedLM.from_pretrained(bert_path, config=config_atk).to(self.device) 
        self.bias_image_feat = None
        self.itm_model = itm_model
        self.itm_model.eval()
        self.itm_model.to(self.device)
        ### bias_image
        self.exp_file = exp_file
        self.bias_image_root = self.exp_file + "/bias_image"
        ### VQG_model
        self.vqg_processor = LlavaNextProcessor.from_pretrained("MODEL/llava-v1.6-mistral-7b-hf")
        self.vqg_model = LlavaNextForConditionalGeneration.from_pretrained("MODEL/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
        self.vqg_model.to(self.device)
        ### declaration
        self.declaration_file = 'BLIP_VQA/___result/DPT/declaration/vqa/val2014_declarative.json'
        with open(self.declaration_file, 'r') as f:
            self.declarations = json.load(f)

        
    def Gen_ori_feats(self, batch):
        image=batch['image'].to(self.device, non_blocking=True)
        img_feats_list,txt_feats_list = self.white_model.Gen_feats(image,batch['question'][0])
        img_feats=torch.cat(img_feats_list, axis=0)
        txt_feats=torch.cat(txt_feats_list, axis=0)
        return img_feats,txt_feats
    

    def pgd_attack(self,x):
        img_feats_list, txt_feats_list = self.white_model.Gen_feats(x, self.batch['question'])
        bias_feats_list, _ = self.white_model.Gen_feats(self.batch['bias_image'], self.batch['question'])
        img_feats = torch.cat(img_feats_list, axis=0)
        txt_feats = torch.cat(txt_feats_list, axis=0)
        bias_feats = torch.cat(bias_feats_list, axis=0)
        return [txt_feats,img_feats,bias_feats]
    

    def _tokenize(self, seq, tokenizer):
        seq = seq.replace('\n', '').lower()
        words = seq.split(' ')
        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)
        return words, sub_words, keys
    

    def get_bpe_substitues(self, substitutes, tokenizer, mlm_model):
        substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates
        all_substitutes = []
        for i in range(substitutes.size(0)):
            if len(all_substitutes) == 0:
                lev_i = substitutes[i]
                all_substitutes = [[int(c)] for c in lev_i]
            else:
                lev_i = []
                for all_sub in all_substitutes:
                    for j in substitutes[i]:
                        lev_i.append(all_sub + [int(j)])
                all_substitutes = lev_i
        c_loss = torch.nn.CrossEntropyLoss(reduction='none')
        word_list = []
        all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
        all_substitutes = all_substitutes[:24].to(self.device)
        N, L = all_substitutes.size()
        word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
        ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))  # [ N*L ]
        ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
        _, word_list = torch.sort(ppl)
        word_list = [all_substitutes[i] for i in word_list]
        final_words = []
        for word in word_list:
            tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
            text = tokenizer.convert_tokens_to_string(tokens)
            final_words.append(text)
        return final_words
    

    def get_substitues(self, substitutes, tokenizer, mlm_model, substitutes_score=None, use_bpe=True, threshold=0.3):
        words = []
        sub_len, k = substitutes.size()  # sub-len, k

        if sub_len == 0:
            return words

        elif sub_len == 1:
            for (i, j) in zip(substitutes[0], substitutes_score[0]):
                if threshold != 0 and j < threshold:
                    break
                words.append(tokenizer._convert_id_to_token(int(i)))
        else:
            if use_bpe == 1:
                words = self.get_bpe_substitues(substitutes, tokenizer, mlm_model)
            else:
                return words
        return words
    def get_important_scores(self, words,batch,tgt_pos,score,image):
        masked_words = self._get_masked(words)
        texts = [' '.join(words) for words in masked_words]
        important_scores = []
        for mlm in texts:
            _,topk_ids, topk_probs = self.black_model(image, mlm, self.answer_candidates, train=False,inference='rank', k_test=128)
            _, pred = topk_probs[0].max(dim=0)
            # print(tgt_pos)
            if tgt_pos not in list(topk_ids[0].cpu().numpy()):
                important_scores.append((torch.tensor(-10000).to(self.device)).data.cpu().numpy())
            else:
                # print(topk_probs[0], [torch.where(topk_ids[0] == tgt_pos)])
                im_value=topk_probs[0][torch.where(topk_ids[0] == tgt_pos)][0]
                important_scores.append((im_value - score).data.cpu().numpy())
        return np.array(important_scores)
    

    def bert_attack(self,batch,tgt_pos,score,gth):
        # self.k=10
        ori_text=batch['question'][0]
        image=batch['image']
        image = image.to(self.device, non_blocking=True)
        text=ori_text.lower()
        feature = Feature(text)
        tokenizer = self.tokenizer_mlm
        words, sub_words, keys = self._tokenize(feature.seq, tokenizer)
        max_length = 512
        inputs = tokenizer.encode_plus(feature.seq, None, add_special_tokens=True, max_length=max_length,
                                       truncation=True)
        input_ids, _ = torch.tensor(inputs["input_ids"]), torch.tensor(inputs["token_type_ids"])
        sub_words = ['[CLS]'] + sub_words[:2] + sub_words[2:max_length - 2] + ['[SEP]']
        input_ids_ = torch.tensor([tokenizer.convert_tokens_to_ids(sub_words)])
        word_predictions = self.mlm_model(input_ids_.to(self.device))[0].squeeze()  # seq-len(sub) vocab
        word_pred_scores_all, word_predictions = torch.topk(word_predictions, self.k, -1)
        word_predictions = word_predictions[1:len(sub_words) + 1, :]
        word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]
        important_scores = self.get_important_scores(words,batch,tgt_pos,score,image)
        feature.query += int(len(words))
        list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=False)
        final_words = copy.deepcopy(words)
        success = 0
        simout = 1
        text_bank=[]
        sim_list=[]
        for ii,top_index in enumerate(list_of_index):
            if feature.change >= self.text_budget:
                feature.success = 1  # exceed
                break
            tgt_word = words[top_index[0]]
            if tgt_word in filter_words:
                continue
            if keys[top_index[0]][0] > max_length - 2:
                continue
            substitutes = word_predictions[keys[top_index[0]][0]:keys[top_index[0]][1]]
            word_pred_scores = word_pred_scores_all[keys[top_index[0]][0]:keys[top_index[0]][1]]
            substitutes = self.get_substitues(substitutes, tokenizer, self.mlm_model, substitutes_score=word_pred_scores)
            most_gap = 0.0
            candidate = None
            distance = []
            for substitute in substitutes:
                if substitute == tgt_word:
                    continue  # filter out original word
                if '##' in substitute:
                    continue  # filter out sub-word
                if substitute in filter_words:
                    continue
                temp_replace = copy.deepcopy(final_words)
                temp_replace[top_index[0]] = substitute
                temp_text = tokenizer.convert_tokens_to_string(temp_replace)
                embs=self.USE_model([ori_text,temp_text]).numpy()
                norm = np.linalg.norm(embs, axis=1)
                embs = embs / norm[:, None]
                sim = (embs[:1] * embs[1:]).sum(axis=1)[0]
                if sim>self.cos_sim:
                    sim_list.append(sim)
                    text_bank.append(temp_text)
                    answer_ids, topk_ids, topk_probs, = self.black_model(image, temp_text,
                                                                         self.answer_candidates, train=False,
                                                                         inference='rank',
                                                                         k_test=128)
                    result = []
                    for ques_id, answer_id in zip(batch['question_id'], answer_ids):
                        result.append({"question_id": int(ques_id.item()), "answer": self.answer_list[answer_id]})
                    ans_after_attack=result[0]['answer']
                    if ans_after_attack != gth:
                        success=1
                        return text_bank,success,sim_list
        text_cand=[]
        if len(text_bank)!=len(sim_list):
            print('wrong bank')
            raise ValueError
        if len(text_bank)!=0:
            sim_list_sort=copy.deepcopy(sim_list)
            for i in range(len(sim_list_sort)):
                si=sim_list_sort.index(max(sim_list_sort))
                text_cand.append(text_bank[si])
                sim_list_sort[si]=-1e8
        return text_cand,success,sim_list
    

    def _get_masked(self, words):
        len_text = max(len(words), 2)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[MASK]'] + words[i + 1:])
        return masked_words
    

    def black_box_predict(self,image,text):
        answer_ids, topk_ids, topk_probs, = self.black_model(image, text,
                                                             self.answer_candidates, train=False,
                                                             inference='rank',
                                                             k_test=128)
        out_v = []
        for  answer_id in answer_ids:
            out_v.append({"answer": self.answer_list[answer_id]})
        return out_v[0]['answer']
 
    def load_bias_image(self,image_path):
        # 读取PIL图像
        image = Image.open(image_path).convert('RGB')
        # 应用转换
        image_tensor = transform(image)  # shape: [3, 480, 480]
        image_tensor = image_tensor.unsqueeze(0)  # shape: [1, 3, 480, 480]
        image_tensor = image_tensor.to(self.device)
        # 归一化操作在GPU上进行
        image_tensor = normalize(image_tensor[0]).unsqueeze(0)

        self.batch["bias_image"] = image_tensor
        return image_tensor


    def get_masked_token_probs(self,sentence, masked_index):
        inputs = self.tokenizer_mlm(sentence, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        with torch.no_grad():
            outputs = self.mlm_model(input_ids=input_ids)
        logits = outputs.logits
        softmax = F.softmax(logits, dim=-1)
        return softmax[0, masked_index, :]


    def VQG_adversarial_generate(self,question,gth,wrong_ans,image_path,ori_img):
        image = Image.open(image_path).convert('RGB')
        if gth.strip().lower().startswith('yes') or gth.strip().lower().startswith('no') or question.strip().lower().startswith('how many'):
            sent = self.declarations[str(int(self.batch['question_id'][0]))]
            gth_sent = sent.replace('[MASK]', gth, 1)
            prompt = "[INST]<image>\n generate atmost questions whose answers are { "+ gth_sent +" , " + wrong_ans +" }, keeping it similar to " + question+ " [/INST] "
        # prompt = "[INST] <image>\nGive a question which {}" + gth + " , " + wrong_ans + "} both answers [/INST]"
        else:
            prompt = "[INST]<image>\n generate atmost questions whose answers are { "+ gth +" , " + wrong_ans +" }, keeping it similar to " + question+ " [/INST] "
        inputs = self.vqg_processor(prompt, image, return_tensors="pt").to(self.device)
        output = self.vqg_model.generate(**inputs, max_new_tokens=100,temperature=ablation_temperature)
        adv_question = self.vqg_processor.decode(output[0], skip_special_tokens=True)
        cand_sent = Split_answer(adv_question,"generate")
        proper_list=[]
        USE_sim_all = 0
        invisile_all = 0
        VLM_sim_all = 0
        for j in range(len(cand_sent)):
            adv_question = cand_sent[j]
            invisile_ans = self.black_box_predict(ori_img,adv_question)
            if invisile_ans == gth:
                embs=self.USE_model([question,adv_question]).numpy()
                norm = np.linalg.norm(embs, axis=1)
                embs = embs / norm[:, None]
                USE_sim = (embs[:1] * embs[1:]).sum(axis=1)[0]
                # USE_sim_all +=USE_sim

                prompt = """[INST]<image>\n The similarity score for two sentences is in the range from 0.0 to 1.0,0.0 means completely different and 1.0 means almost the same. 
                Now given two sentences \" """ + question +"\" and \"" + adv_question + """\", please give a similarity score for these two sentences: The similarity score for these two sentences is [/INST] """
                inputs = self.vqg_processor(prompt, image, return_tensors="pt").to(self.device)
                output = self.vqg_model.generate(**inputs, max_new_tokens=100,temperature=ablation_temperature)
                VQG_sim = self.vqg_processor.decode(output[0], skip_special_tokens=True)
                VQG_sim = Split_answer(VQG_sim,"sim")
                if VQG_sim >= therehold_for_aisim or USE_sim >= 0.95:
                    invisile_all += 1         
                    proper_list.append(adv_question)
        return proper_list
    
    def get_mask_text(self,ori_text):
        text=ori_text.lower()
        feature = Feature(text)
        tokenizer = self.tokenizer_mlm
        words, sub_words, keys = self._tokenize(feature.seq, tokenizer)
        masked_words = self._get_masked(words)
        masked_text = [' '.join(words) for words in masked_words]
        return masked_text,words

    def VQG_adversarial_mask(self,question,gth,wrong_ans,image_path,ori_img):
        image = Image.open(image_path).convert('RGB')
        masked_text,words = self.get_mask_text(question)
        cand_sent = []
        for j in range(len(masked_text)):
            prompt = "[INST]<image>\n Only replace all [MASK] tokens in \""+ masked_text[j] +"\" to generate a new sentence.Do not use the word "+ words[j] +" in the replacement. [/INST]"
            inputs = self.vqg_processor(prompt, image, return_tensors="pt").to(self.device)
            output = self.vqg_model.generate(**inputs, max_new_tokens=100,temperature=ablation_temperature)
            adv_question = self.vqg_processor.decode(output[0], skip_special_tokens=True)
            cand_sent_temp = Split_answer(adv_question,"mask")
            cand_sent.append(cand_sent_temp)
        proper_list=[]
        invisile_all = 0
        for j in range(len(cand_sent)):
            adv_question = cand_sent[j]
            invisile_all += 1

            embs=self.USE_model([question,adv_question]).numpy()
            norm = np.linalg.norm(embs, axis=1)
            embs = embs / norm[:, None]
            USE_sim = (embs[:1] * embs[1:]).sum(axis=1)[0]

            prompt = """[INST]<image>\n The similarity score for two sentences is in the range from 0.0 to 1.0,0.0 means completely different and 1.0 means almost the same. 
            Now given two sentences \" """ + question +"\" and \"" + adv_question + """\", please give a similarity score for these two sentences: The similarity score for these two sentences is [/INST] """
            inputs = self.vqg_processor(prompt, image, return_tensors="pt").to(self.device)
            output = self.vqg_model.generate(**inputs, max_new_tokens=100,temperature=ablation_temperature)
            VLM_sim = self.vqg_processor.decode(output[0], skip_special_tokens=True)
            VLM_sim = Split_answer(VLM_sim,"sim")
            if VLM_sim >= therehold_for_aisim or USE_sim >= 0.95:         
                proper_list.append(adv_question)
        return proper_list




    @torch.no_grad()
    def evaluate(
        self,
        data_loader,
        tokenizer
    ):
        answer_list = data_loader.dataset.answer_list
        self.answer_list=answer_list
        answer_candidates = self.black_model.tokenizer(answer_list, padding='longest', return_tensors='pt').to(self.device)
        self.answer_candidates=answer_candidates
        answer_candidates.input_ids[:, 0] = self.black_model.tokenizer.bos_token_id
        self.tokeizer=tokenizer
        self.white_model.eval()
        self.black_model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Test:"
        print_freq=500
        json_file_path = 'BLIP_VQA/___result/temperature/' + str(ablation_temperature) + '.json' 
        list=[]
        lens=0
        for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            if len(self.acc_list)>=5000:
                break
            if int(batch['question_id'][0]) not in self.correct_list:
                continue
            ori_img = batch['image'].to(self.device, non_blocking=True)
            pred_ans = self.black_box_predict(ori_img,batch['question'][0])

            ori_img = batch['image'].to(self.device, non_blocking=True)
            self.batch = copy.deepcopy(batch)
            bias_img = self.load_bias_image(self.bias_image_root + '/'+ str(batch['question_id'][0].cpu().numpy()) + '.png')

            ret = dict()
            ret['preds'] = [self.blip_ans_table[str(int(batch['question_id'][0]))]]
            if pred_ans!=ret['preds'][0]:
                print('wrong answer here',pred_ans,ret['preds'][0])
                continue

            wrong_ans = self.black_box_predict(bias_img,batch['question'][0])
            adv_question_list_mask  = self.VQG_adversarial_mask(batch['question'][0],pred_ans,wrong_ans,batch['image_path'][0],ori_img)
            adv_question_list  = self.VQG_adversarial_generate(batch['question'][0],pred_ans,wrong_ans,batch['image_path'][0],ori_img)

            self.acc_list.append(torch.tensor(1,device=self.device))
            question_dict = {  
                    "question_id": str(int(batch['question_id'][0])),  
                    "ori_question": batch['question'][0],  
                    "adv_question": adv_question_list_mask + adv_question_list 
                }
            list.append(question_dict)
            lens += len(adv_question_list)
            
        with open(json_file_path, 'w', encoding='utf-8') as file:  
            json.dump(list, file, ensure_ascii=False, indent=4)    
        print("avg count:",lens/len(self.acc_list))
        print("complete!")

import torch
import numpy as np
import matplotlib.pyplot as plt
import re
 

def Split_answer(ques_all,flag):
    index  = ques_all.find('[/INST]')
    ques_all = ques_all[index + len('[/INST] '):]
    # questions = [q.strip().split('. ', 1)[1] for q in ques_all.split('\n')]
    if flag == "mask":
        return ques_all
    if flag == "generate":
        questions = []
        for q in ques_all.split('\n'):
            parts = q.strip().split('. ', 1)
            if len(parts) > 1:
                questions.append(parts[1])
        return questions
    if flag == "sim":
        match = re.search(r'(\d+\.\d+)', ques_all)
        if match:        
            return float(match.group(1))
        else:
            return 0.0
    return []
