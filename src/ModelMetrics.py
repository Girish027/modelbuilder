#!/usr/bin/env python
# coding: utf-8
##script created using python 3.6
## Aim of this script is to create metrics from the results obtained by passing the output from sklearn CV experiment
# through the sigmoid function using params learned using workbench
import pandas as pd
import numpy as np
import os,sys,shutil
import json
import configparser
import pickle,bz2
from sklearn.metrics import f1_score,accuracy_score,precision_score,precision_recall_curve,classification_report,confusion_matrix
from collections import defaultdict
import copy

def sigmoid_predict(d,score):
    a = d['a']
    b = d['b']
    x=score*a+b
    return 1/(1 + np.exp(x))

def get_nbest(scores,maxintents,classes):
    t = np.array(scores)
    # sorting & fetching index in descending order of values
    #(::-1 is for inversing & making it descending order)
    indices = t.argsort()[::-1][:maxintents]
    top_classes = classes[indices]
    top_scores = t[indices]
    return [{'className':c,'score':s} for c,s in zip(top_classes,top_scores)]

def create_flat_dict(dict_list):
    d = {}
    for i,out in enumerate(dict_list):
        d.update({'Classified Intent {}'.format(i+1):out['className'],'Classification Score {}'.format(i+1):out['score']})
    return d

def get_score(x,sorted_sigmoid,sklearn_model):
    scores = sklearn_model.decision_function([x['embedding']])[0]
    if(len(sklearn_model.classes_) == 2):
        scores_new = [scores,-1*scores]
        scores = scores_new
    scaled_scores = [sigmoid_predict(sorted_sigmoid[i][1],score) for i,score in enumerate(scores)]
    nbest = get_nbest(scaled_scores,2,sklearn_model.classes_)
    d = create_flat_dict(nbest)
    d['filename'] = x['filename']
    x = pd.Series(d)
    return x

def get_output_datafram_workbench_format(df_complete,df_test):
    columns_to_pick = list(df_complete.columns)
    ## removing embedding column
    columns_to_pick.remove('embedding')
    df_complete = df_complete[columns_to_pick]
    ## inner join or right join would be fine
    t = df_complete.merge(df_test,on='filename',how='inner')
    t['Diff between Top Intents'] = t['Classification Score 1'] - t['Classification Score 2']
    t['Recognition Source'] = 'SLM'
    t['Correct Classification'] = t['intent']==t['Classified Intent 1']
    return t

def save_workbench_format_file(df_out,sklearn_folder,output_filename):
    output_path = os.path.join(sklearn_folder,output_filename)
    df_out.to_csv(output_path,sep=',',header=True,index=False)

def parse_sigmoid_params_and_generate_classification_out(df_complete,input_folder,sklearn_folder,fold_index, report_types):
    unique_id_path = os.path.join(input_folder,'unique_ids.txt')
    df_unique_id = pd.read_csv(unique_id_path,sep='\t',header=None,names=['filename'])
    sigmoid_file = os.path.join(input_folder,"sigmoidParams.csv")
    df_sigmoid = pd.read_csv(sigmoid_file,sep=',')
    dict_sigmoid = df_sigmoid.set_index('intent').to_dict(orient='index')
    sorted_sigmoid = sorted(dict_sigmoid.items())
    sklearn_model_path = os.path.join(input_folder,'sklearn_model.bz2')
    with bz2.BZ2File(sklearn_model_path,'r') as rf:
        sklearn_saved_model = pickle.load(rf)

    df_test = df_unique_id.merge(df_complete,on='filename',how='left')
    df_test = df_test.apply(lambda x: get_score(x,sorted_sigmoid,sklearn_saved_model),axis=1)
    df_out = get_output_datafram_workbench_format(df_complete,df_test)
    if(len(df_out)>0):
        conf_labels = np.unique(df_out['intent'])
        conf_matrix = pd.DataFrame(confusion_matrix(df_out['intent'],df_out['Classified Intent 1']),columns=conf_labels,index=conf_labels)
        conf_matrix_file = os.path.join(sklearn_folder,f'ConfusionMatrix_{fold_index+1}.csv')
        save_report(conf_matrix,conf_matrix_file)
    output_per_report_type = {}
    for report_type, row_types in report_types.items():
        df_out_filtered = df_out[df_out['row_type'].isin(row_types)]
        save_workbench_format_file(df_out_filtered,sklearn_folder,f'ClassificationOutput_{report_type}_{fold_index+1}.csv')
        if(len(df_out_filtered)>0):
            conf_labels = np.unique(df_out_filtered['intent'])
            conf_matrix = pd.DataFrame(confusion_matrix(df_out_filtered['intent'],df_out_filtered['Classified Intent 1']),columns=conf_labels,index=conf_labels)
            FP = (conf_matrix.sum(axis=0) - np.diag(conf_matrix) ).tolist()
            FN = (conf_matrix.sum(axis=1) - np.diag(conf_matrix)).tolist()
            TP = np.diag(conf_matrix).tolist()
            conf_matrix_file = os.path.join(sklearn_folder,f'ConfusionMatrix_{report_type}_{fold_index+1}.csv')
            save_report(conf_matrix,conf_matrix_file)
        cr = get_classification_report(df_out_filtered)
        cr['TP']=0
        cr['FP']=0
        cr['FN']=0
        if(len(df_out_filtered)>0):
            list_place_holder = [0,0,0]
            cr['TP'] = TP + list_place_holder
            cr['FP'] = FP + list_place_holder
            cr['FN'] = FN + list_place_holder
        cr_out_file = os.path.join(sklearn_folder,f'ClassificationReport_{report_type}_{fold_index+1}.csv')
        save_report(cr,cr_out_file)
        # removing the row_type column, this is not necessary. Just wanted to hide this info especially for external user
        df_out_filtered = df_out_filtered.drop('row_type',axis=1)
        cr = cr.drop(['TP','FP','FN'],axis=1)
        output_per_report_type[report_type] = {'df_foldwise':df_out_filtered, 'classification_report_foldwise' : cr}
    return output_per_report_type

def get_classification_report(df):
    c  = {}
    default_value = {'precision': np.nan, 'recall': np.nan, 'f1-score': np.nan, 'support': 0}
    c['micro avg'] = copy.deepcopy(default_value)
    c['macro avg'] = copy.deepcopy(default_value)
    c['weighted avg'] = copy.deepcopy(default_value)
    if(len(df)>0):
        c = classification_report(df['intent'],df['Classified Intent 1'],output_dict=True)

        if "accuracy" in c:
            # if all labels are used while generating classification report for a multiclass setting,then
            # micro avg is the same for precision, recall & hence f1-score. In newer version of sklearn instead of
            # micro average result is kept as accuracy. Changing that back to micro-average so that it is easier to
            # load & save with pandas
            a = c.get("accuracy")
            del c['accuracy']
            support = c.get("macro avg").get("support")
            c['micro avg'] = {'precision': a, 'recall': a, 'f1-score': a, 'support': support}
    c = pd.DataFrame.from_dict(c,orient='index')
    c['intent frequency']=c['support']/len(df) * 100
    ## keeping the average values at the end
    avg_indexes = ['micro avg','macro avg','weighted avg']
    indexes = c.index.tolist()
    for ri in avg_indexes:
        indexes.remove(ri)
    c = c.reindex(indexes+avg_indexes)
    c.index.name = "Intents"
    return c

def save_report(df,filename):
    df.to_csv(filename,sep=',',header=True,index=True)

def calculate_save_avg_report(l,filename):
    cr_combined = pd.concat(l,axis=1)
    cr_g = cr_combined.groupby(cr_combined.columns,axis=1)
    cr_a = cr_g.agg(np.mean)
    cr_std = cr_g.agg(np.std)
    cr_out = cr_a.merge(cr_std,left_index=True,right_index=True,suffixes=("_average","_std"))
    rearranged_columns =  sorted(cr_out.columns)
    cr_out = cr_out[rearranged_columns]
    save_report(cr_out,filename)


def cleanup_files(sklearn_folder):
    folds_folder = os.path.join(sklearn_folder,'folds')
    shutil.rmtree(folds_folder)
    files_to_cleanup = ['data.txt', 'bias_out.txt','classes.txt','coeff_out.txt','features_out.txt','pred_out.txt','sigmoidParams.txt']
    for f in files_to_cleanup:
        os.remove(os.path.join(sklearn_folder,f))
