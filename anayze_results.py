import csv
import json
from statsmodels.stats.descriptivestats import sign_test
from statsmodels.stats.inter_rater import fleiss_kappa
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

models = ['clipcap', 'mplug', 'git', 'vit_gpt2']
datasets = ['coco', 'flickr', 'xm3600']

def analyze_human_evaluation(res_files):
    res = []
    for res_file in res_files:
        with open(res_file, 'r') as fp:
            my_reader = csv.reader(fp)
            cur_res = list(my_reader)
            cur_res = cur_res[1:]
            res.append(cur_res)
    cur_ind = 0
    question_num = 5
    results = {}
    for model in models:
        results[model] = {}
        for dataset in datasets:
            results[model][dataset] = {'re': [0]*question_num, 'base': [0]*question_num, 'tie': [0]*question_num}
            with open(f'{model}_{dataset}_data.json', 'r') as fp:
                data = json.load(fp)
            for i in range(len(data)):
                for question_ind in range(question_num):
                    if len([x for x in res if x[cur_ind*3][3 + question_ind].startswith('1')]) > 1:
                        if data[i]['captions'][0][0] == 're':
                            results[model][dataset]['re'][question_ind] += 1
                        else:
                            results[model][dataset]['base'][question_ind] += 1
                    elif len([x for x in res if x[cur_ind*3][3 + question_ind].startswith('2')]) > 1:
                        if data[i]['captions'][1][0] == 're':
                            results[model][dataset]['re'][question_ind] += 1
                        else:
                            results[model][dataset]['base'][question_ind] += 1
                    else:
                        results[model][dataset]['tie'][question_ind] += 1
                cur_ind += 1
				
    return results

def compute_annotator_agreement(res_files):
    res = []
    for res_file in res_files:
        with open(res_file, 'r') as fp:
            my_reader = csv.reader(fp)
            cur_res = list(my_reader)
            cur_res = cur_res[1:]
            res.append(cur_res)
    
    sample_num = int(len(res[0])/3)
    question_num = 5
    res_mats = []
    for _ in range(question_num):
        res_mats.append(np.zeros((sample_num, 3)))

    for i in range(sample_num):
        for question_ind in range(question_num):
            for j in range(len(res)):
                if res[j][i*3][3 + question_ind].startswith('1'):
                    res_mats[question_ind][i, 0] += 1
                elif res[j][i*3][3 + question_ind].startswith('2'):
                    res_mats[question_ind][i, 1] += 1
                else:
                    res_mats[question_ind][i, 2] += 1
				
    return [fleiss_kappa(x) for x in res_mats]

def run_sign_test(results, model, dataset):
    question_num = 5
    all_re = [0]*question_num
    all_base = [0]*question_num
    all_tie = [0]*question_num

    if model is not None:
        models = [model]

    if dataset is not None:
        datasets = [dataset]

    for model in models:
        for dataset in datasets:
            for i in range(5):
                all_re[i] += results[model][dataset]['re'][i]
                all_base[i] += results[model][dataset]['base'][i]
                all_tie[i] += results[model][dataset]['tie'][i]

    p_values = []
    for i in range(question_num):
        p_values.append(sign_test([1]*all_re[i]+[-1]*all_base[i])[1])

    return p_values

def plot_results(results, model, dataset):
    question_num = 5
    all_re = [0]*question_num
    all_base = [0]*question_num
    all_tie = [0]*question_num

    if model is not None:
        models = [model]

    if dataset is not None:
        datasets = [dataset]

    for model in models:
        for dataset in datasets:
            for i in range(5):
                all_re[i] += results[model][dataset]['re'][i]
                all_base[i] += results[model][dataset]['base'][i]
                all_tie[i] += results[model][dataset]['tie'][i]

    font = {'size' : 16}
    matplotlib.rc('font', **font)

    questions = (
        'Faithfulness',
        'Completeness',
        'Accuracy',
        'Detail',
        'Overall',
    )
    question_num = len(questions)

    # Sort plot by how much we won
    orig_ind = [(i, all_re[i]) for i in range(question_num)]
    orig_ind.sort(key=lambda x:x[1], reverse=True)
    new_to_orig_inds = [x[0] for x in orig_ind]
    questions = [questions[new_to_orig_inds[i]] for i in range(question_num)]
    weight_counts = {
        "base+re": np.array([all_re[new_to_orig_inds[i]]/2 for i in range(question_num)]),
        "Tie": np.array([all_tie[new_to_orig_inds[i]]/2 for i in range(question_num)]),
        "base": np.array([all_base[new_to_orig_inds[i]]/2 for i in range(question_num)]),
    }

    width = 0.5

    fig, ax = plt.subplots()
    left = np.zeros(question_num)
    colors = [(0, 0.4, 1), (0.298, 0.5843, 0.9843), (0.698, 0.8196, 1)]
    y_pos = [0.6*i for i in range(question_num)]
    x_ticks = [0, 25, 50, 75, 100]

    ind = 0
    for source, weight_count in weight_counts.items():
        p = ax.barh(y_pos, weight_count, width, label=source, left=left, color=colors[ind])
        ax.set_yticks(y_pos, labels=questions)
        ax.set_xticks(x_ticks, labels=[f'{x}%' for x in x_ticks])
        if source == 'base+re':
            for i in range(question_num):
                plt.text((weight_count[i]-10)/2, -0.1+0.6*i, f'*{int(weight_count[i])}%', color='white', fontsize=16)
        elif source == 'Tie':
            for i in range(question_num):
                plt.text(weight_counts['base+re'][i]+(weight_count[i]-10)/2, -0.1+0.6*i, f'{int(weight_count[i])}%', color='black', fontsize=16)
        else:
            for i in range(question_num):
                base_loc = weight_counts['base+re'][i]+weight_counts['Tie'][i]+(weight_count[i]-10)/2
                if i == 1:
                    plt.text(base_loc+3, -0.1+0.6*i, f'{int(weight_count[i])}%', color='black', fontsize=16)
                elif i == 2:
                    plt.text(base_loc+2, -0.1+0.6*i, f'{int(weight_count[i])}%', color='black', fontsize=16)
                elif i == 3:
                    plt.text(base_loc+3, -0.1+0.6*i, f'{int(weight_count[i])}%', color='black', fontsize=16)
                elif i == 4:
                    plt.text(base_loc+3, -0.1+0.6*i, f'{int(weight_count[i])}%', color='black', fontsize=16)
                else:
                    plt.text(base_loc, -0.1+0.6*i, f'{int(weight_count[i])}%', color='black', fontsize=16)
        ind += 1
        left += weight_count

    ax.legend(loc='center left', bbox_to_anchor=(0.01, 1.07), ncols=3)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(8, 4)
    plt.subplots_adjust(left=0.25, right=0.9, top=0.85, bottom=0.1)

    plt.savefig('human_eval.png')
