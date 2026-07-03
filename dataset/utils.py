import re
import json
import os
import torch
import utils

def pre_question(question,max_ques_words):
    question = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        question.lower(),
    ).replace('-', ' ').replace('/', ' ')  
    question = question.rstrip(' ')
    
    #truncate question
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
            
    return question
        
def save_result(result, result_dir, filename, is_json=True, is_list=True, remove_duplicate=""):
    if is_json:
        result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
        final_result_file = os.path.join(result_dir, '%s.json'%filename)
        json.dump(result,open(result_file,'w'))
    else:
        result_file = os.path.join(result_dir, '%s_rank%d.pth'%(filename,utils.get_rank()))
        final_result_file = os.path.join(result_dir, '%s.pth'%filename)
        torch.save(result,result_file)     

    if is_list:
        result = []
    else:
        result = {}
    for rank in range(utils.get_world_size()):
        if is_json:
            result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
            res = json.load(open(result_file,'r'))
        else:
            result_file = os.path.join(result_dir, '%s_rank%d.pth'%(filename,rank))
            res = torch.load(result_file)            
        if is_list:
            result += res
        else:
            result.update(res)
    if remove_duplicate:
        result_new = []
        id_list = []    
        for res in result:
            if res[remove_duplicate] not in id_list:
                id_list.append(res[remove_duplicate])
                result_new.append(res)
        result = result_new  
    if is_json:                  
        json.dump(result,open(final_result_file,'w'))   
    else:            
        torch.save(result,final_result_file)     
    
    print('result file saved to %s'%final_result_file)
    return final_result_file
