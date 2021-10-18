import codecs
import concurrent.futures
import csv
import math
import itertools
import json
from itertools import chain
from collections import defaultdict
import Task
import zipfile
import ModelMetrics
import logging
import os
from sklearn.metrics import f1_score,accuracy_score,classification_report,confusion_matrix
import numpy as np
import pandas as pd
import pygogo as gogo
import requests
import linearsvc
import time
import fnmatch
import logging
import pygogo as gogo
from config import data

log_format = '[%(levelname)s] %(asctime)s: [%(modeluuid)s] %(message)s'
date_format = '%Y-%m-%dT%H:%M:%S%z'
formatter = logging.Formatter(log_format, date_format)

class Normalizer:
    """This class submit job to web2nl executor for normalizing via web2nl and than write to a file to be
    submitted to DSG Java Workbench"""
    # No of worker threads to be used for sending requests to web2nl
    NO_OF_WORKER_THREADS = data.get("NO_OF_WORKER_THREADS",8)
    def __init__(self, modelUUID, logger, default_intent='None_None', tfClient=None, modelTechnology= None, modelName=None, vectorizerVersion=None, normalizationUUID="Norm_test"):
        self.tfx_url = data.get("TFX_URL", None)
        self.tfx_url_key = data.get("TFX_URL_KEY", None)
        if self.tfx_url is None or self.tfx_url_key is None:
            logger.error("TFX URL or APi KEY not found")
            raise ValueError
        logger.info("Setting TFX URL: %s", self.tfx_url) # Base url /v1/machinelearning/predictions 
        self.model_uuid = modelUUID
        self.logger = logger
        self.tfClient = tfClient
        self.modelTechnology = modelTechnology
        self.modelName = modelName
        if (modelTechnology == 'use' or modelTechnology == 'use_large'):
            self.tfClient = 'standard'
            self.modelName = 'model'
        self.vectorizerVersion = vectorizerVersion
        self.transcript_mappings = pd.DataFrame(columns=["transcript","embedding"])
        self.intent_mappings = {}
        self.normalize_text_mapping = {}
        logger.info("Normalization handler initialized successfully")
        self.default_intent = default_intent
        logger.info("Setting default intent to: %s", self.default_intent)

    def get_default_intent(self):
        return self.default_intent

    def get_embUrl(self):
        return f"/clients/{self.tfClient}/applications/{self.modelTechnology}/models/{self.modelName}/labels/{self.vectorizerVersion}"

    def get_embeddings(self, input_string):
        """Normalizes data using web2nl"""
        try:
            # todo add batch calls by passing multiple text inputs -- DONE (Embedding values of same string varies
            #  by providing it in the batch mode. Need to verify which approach should be used)
            # todo try to take unique text before passing it to tfx call -- DONE
            tfx_start = time.time()
            payload = {
                "signature_name": "serving_default",
                "inputs": {
                  "input_text":input_string
                }
            }
            body = json.dumps(payload)
            api_endpoint = self.tfx_url + f"/clients/{self.tfClient}/applications/{self.modelTechnology}/models/{self.modelName}/labels/{self.vectorizerVersion}"
            response = requests.post(url = api_endpoint, data=body, headers = {"Content-Type": "application/json","apikey": self.tfx_url_key})
            response.raise_for_status()
            response_in_json = response.json()
            text = response_in_json.get("outputs", None)
            tfx_end = time.time()
            self.logger.info("TFX Call: %s", str(tfx_end - tfx_start))
            if text is None:
                raise RuntimeError("Failed getting normalized form from web2nl for: " + input_string)
            return text
        except Exception as err:
            print("Exception thrown for input : ",input_string)
            raise err

    def fetch(self, transcription=[]):
        """Calling web2nl to get normalized text and than write to a file"""
        df = pd.DataFrame(transcription,columns=["transcripts"])
        transcript_list = df.transcripts.unique().tolist()
        normalized_text = self.get_embeddings(transcript_list)
        embedding_list = [str(ele).strip('[').strip(']') for ele in normalized_text]
        res_df = pd.DataFrame(list(zip(transcript_list,embedding_list)),columns=["transcript","embedding"])
        return res_df

    def getEmbeddings(self, workDirectory):
        utterance_count = 0
        try:
            self.logger.info("Starting to normalize data via web2nl. It may take some time depending on data size")
            training_data_file = None
            for file in os.listdir(os.path.join(workDirectory, "data")):
                if fnmatch.fnmatch(file, 'input*'):
                    training_data_file = os.path.join(os.path.join(workDirectory, "data"), file)
            df_orig = pd.read_csv(training_data_file,sep='\t',encoding='utf-8')
            df_orig.columns = ["intent", "transcript","Original Transcription","Granular Intent", "filename","Transcription Hash", "Comments","row_type"]
            df_orig = df_orig.drop(["Original Transcription","Granular Intent","Transcription Hash", "Comments"],axis=1)
            df_orig = df_orig.replace(np.NaN,'')
            intent_list_unique = df_orig.intent.unique()
            if self.default_intent not in intent_list_unique:
                for i in range(10):
                    df_extra = {'intent': self.default_intent, 'transcript': 'None none input'+str(i), 'filename': 'utt_dummy_03042019_016'+str(i), 'row_type': 'I'}
                    df_orig = df_orig.append(df_extra, ignore_index = True)
            transcripts_list_unique = df_orig.transcript.unique()
            transcription_list_batch = [[]]
            batch_list_counter = 0
            batch_list_index = 0
            for i in transcripts_list_unique:
                if batch_list_counter == 100:
                        transcription_list_batch.append([])
                        batch_list_counter = 0
                        batch_list_index = batch_list_index + 1
                transcription_list_batch[batch_list_index].append(i)
                batch_list_counter = batch_list_counter + 1
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.NO_OF_WORKER_THREADS) as executor:
                future_map_res = executor.map(self.fetch,transcription_list_batch)
            for res_df in future_map_res:
                self.transcript_mappings = self.transcript_mappings.append(res_df).drop_duplicates(subset='transcript')
            train_df = df_orig.merge(self.transcript_mappings,how='inner',on='transcript')
            training_file_path = os.path.join(workDirectory, "FirstResponses_WCNormalized")
            train_df.to_csv(training_file_path,index=False,sep='\t')
            self.logger.info("Total Utterances: %s", str(utterance_count))
            if len(transcripts_list_unique) is 0:
                self.logger.error("No transcription received for classification")
                raise RuntimeError("No transcription received for classification")
            # if self.default_intent not in self.intent_mappings:
            #     default_transcription = "0.0, 0.0, 0.0, 0.0"
            #     self.logger.warn("Since default intent was not found in training data, adding dummy transcription with "
            #                      "intent: %s", self.default_intent)
            #     self.write_to_file(default_transcription, self.default_intent, "default-file", "I", normalized_file_writer)
            self.logger.info("Training a classifier model with intents: %s", self.intent_mappings)
        except Exception as exc:
            raise exc

# if __name__=="__main__":


def get_all_file_paths(directory):
  
    # initializing empty file paths list
    file_paths = []
  
    # crawling through directory and subdirectories
    for root, directories, files in os.walk(directory):
        for filename in files:
            # join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
  
    # returning all file paths
    return file_paths        

def tfx_normalization(modelUUID, logger, default_intent='None_None', normalizationUUID="Norm_test",tfClient=None,modelTechnology=None,modelName=None,vectorizerVersion=None,postProcessingRules=None):
    start_time = time.time()
    norm_start = time.time()
    # Getting embeddings from TFX service
    workDirectory = os.path.join(data['MODEL_BUILDER_STORAGE'], modelUUID)
    try:
        Task.updateStatus(workDirectory, "Normalizing data via Tensorflow")
        normalizer = Normalizer(modelUUID,logger,default_intent, tfClient,modelTechnology,modelName,vectorizerVersion,normalizationUUID)
        normalizer.getEmbeddings(workDirectory)
        norm_end = time.time()
        logger.info("Normalization time taken : %s",str(norm_end-norm_start))
        Task.updateStatus(workDirectory, "Tensorflow Normalization completed, starting to build model via workbench...")
        # Creating the train data for svm model
        train = []
        intent = []
        training_file_path = os.path.join(workDirectory, "FirstResponses_WCNormalized")
        df_orig = pd.read_csv(training_file_path,sep='\t',encoding='utf-8')
        df_orig['transcript'].to_csv(f"{workDirectory}/corpus_1",header=False,index=False)
        df_orig['embedding'] = [x.split(',') for x in df_orig['embedding']]
        df_orig['embedding'] = df_orig['embedding'].apply(lambda x : pd.to_numeric(x))
        train = df_orig['embedding'].to_list()
        intent = df_orig['intent'].to_list()
        train_split = np.array(train)
        output_label = np.array(intent)
        
        # Setting this seed to make model return same output everytime
        np.random.seed(12345)
        
        # Perfroming grid search to find best params of model
        # todo compare the timings of each step and figure out which call takes more time -- DONE
        #  (Need to perform the same analysis in SV2 once we deploy in lower environments to check network cost)
        # todo see if the cv fits take the same intent order while fitting -- DONE (Ordering is based on internal alphabetical order)
        grid_start = time.time()
        grid_search = linearsvc.perform_grid_search(train_split,output_label)
        grid_end = time.time()
        logger.info("Grid search time taken : %s",str(grid_end-grid_start))
        final_model = grid_search.best_estimator_ # Best model from grid search
        
        # Perform a cv fold to get the svm scores which is passed to sigmoid train as training data
        sigmoid_scores_start = time.time()
        y_pred_scores,y_pred,y_actual,indexes_order = linearsvc.get_scores_for_sigmoid_training(train_split,output_label,final_model.get_params(), 3)
        sigmoid_scores_end = time.time()
        logger.info("Sigmoid scores generation time taken : %s",str(sigmoid_scores_end-sigmoid_scores_start))
        
        # Coverting the svm predicted scores to a df with class names as column names
        y_pred_df = pd.DataFrame(y_pred_scores,columns=grid_search.classes_.tolist())
        
        # Writing the scores to a file
        score_path = os.path.join(workDirectory, "Results")
        os.mkdir(score_path)
        linearsvc.write_sigmoid_scores(y_pred_scores,y_actual,score_path)
        logger.info("Produced sigmoid train data")
        
        # Computing sigmoid train params for each intent
        sigmoid_params = []
        sigmoid_train_start = time.time()
        for idx in range(y_pred_df.columns.size):
            sigmoid_data = y_pred_df.iloc[:,idx].tolist()
            label = []
            for i in range(len(sigmoid_data)):
                # todo compare the column value of intent and row value of intent match then give 1 else 0 -- DONE
                if y_pred_df.columns[idx] == y_actual[i]:
                    label.append(float(1))
                else:
                    label.append(float(0))
            sigmoid_params.append(linearsvc.sigmoid_train(len(sigmoid_data),sigmoid_data,label))
        sigmoid_train_end = time.time()
        logger.info("Sigmoid train time taken : %s",str(sigmoid_train_end-sigmoid_train_start))
        logger.info("Sigmoid Params : %s" , str(sigmoid_params))

        report_types = json.loads('{"internal":["I","A","E"],"external":["E"]}')
        df_combined = defaultdict(pd.DataFrame)
        dfs = defaultdict(list)
        crs = defaultdict(list)
        report_wise_output = linearsvc.perform_cv_for_performance_metrics(train_split,output_label,df_orig,final_model.get_params(), score_path)
        for report in report_wise_output:
            for report_type, row_types in report_types.items():
                    df_foldwise = report[report_type]['df_foldwise']
                    classification_report_foldwise = report[report_type]['classification_report_foldwise']
                    dfs[report_type].append(df_foldwise[['intent','Classified Intent 1']])
                    crs[report_type].append(classification_report_foldwise)
                    df_combined[report_type] = df_combined[report_type].append(df_foldwise)

        f1_total = []
        acc_total = []
        for report_type, row_types in report_types.items():
            ModelMetrics.save_workbench_format_file(df_combined[report_type],os.path.join(workDirectory, "Results"),f'ClassificationOutput_combined_{report_type}.csv')
            avg_f1_weighted = 0
            f1_weighted_std = 0
            avg_acc = 0
            acc_std = 0
            f1s_weighted = [f1_score(df['intent'],df['Classified Intent 1'],average='weighted') for df in dfs[report_type]]
            if(len(f1s_weighted) != 0):
                f1_total = f1_total + f1s_weighted
                avg_f1_weighted = np.mean(f1s_weighted)
                f1_weighted_std = np.std(f1s_weighted)
            logger.info(f'average weighted f1_score for report_type {report_type}', avg_f1_weighted, ' :+- ',f1_weighted_std )
            accuracies = [accuracy_score(df['intent'],df['Classified Intent 1']) for df in dfs[report_type]]
            if(len(accuracies) != 0):
                acc_total = acc_total + accuracies
                avg_acc= np.mean(accuracies)
                acc_std = np.std(accuracies)
            logger.info(f'average accuracy for report_type {report_type}', avg_acc, ' :+- ',acc_std )
            df_actual_vs_predicted = pd.DataFrame()
            for df in dfs[report_type]:
                df = df.loc[df['intent'] != df['Classified Intent 1']]
                df_actual_vs_predicted = df_actual_vs_predicted.append(df)
            df_stats = df_actual_vs_predicted[['intent','Classified Intent 1']].groupby(['intent','Classified Intent 1']).size()
            df_aggregate_metrics = pd.DataFrame.from_dict({'0':{'Metric':'weighted fscore','Average':avg_f1_weighted, 'Std Dev': f1_weighted_std},
                '1':{'Metric':'accuracy','Average':avg_acc, 'Std Dev':avg_f1_weighted}},
                orient='index')
            aggregate_metrics_path = os.path.join(os.path.join(workDirectory, "Results"),f'aggregate_metrics_{report_type}_cumulative.csv')
            stats_path = os.path.join(workDirectory,f"model_stats_{report_type}.xlsx")
            df_aggregate_metrics.to_csv(aggregate_metrics_path,sep=',',header=True,index=False)
            writer = pd.ExcelWriter(stats_path)
            df_aggregate_metrics.to_excel(writer, sheet_name='Sheet1')
            df_stats = df_stats.rename_axis(['Actual Intent','Predicted Intent']).reset_index(name='count')
            if(len(df_stats) > 0):
                df_stats.to_excel(writer, sheet_name='Sheet1', startrow=5)
            writer.save()
            cr_out_file = os.path.join(os.path.join(workDirectory, "Results"),f'ClassificationReport_combined_{report_type}.csv')
            ModelMetrics.calculate_save_avg_report(crs[report_type],cr_out_file)

        f1_total = [i for i in f1_total if i != 0]
        acc_total = [0 if math.isnan(x) else x for x in acc_total]
        acc_total = [i for i in f1_total if i != 0]
        avg_f1_total = np.mean(f1_total)
        f1_total_std = np.std(f1_total)
        avg_acc_total = np.mean(acc_total)
        acc_total_std = np.std(acc_total)
        df_total_aggregate_metrics = pd.DataFrame.from_dict({'0':{'Metric':'weighted fscore','Average':avg_f1_total, 'Std Dev': f1_total_std},
                '1':{'Metric':'accuracy','Average':avg_acc_total, 'Std Dev':acc_total_std}},
                orient='index')
        total_aggregate_metrics_path = os.path.join(f"{workDirectory}/Results",f'aggregate_metrics_cumulative.csv')
        df_total_aggregate_metrics.to_csv(total_aggregate_metrics_path,sep=',',header=True,index=False)
        os.system("cp " + workDirectory + "/model_stats_internal.xlsx "+ workDirectory + "/model_stats.xlsx")
        # Generating web2nl model file
        embUrl = normalizer.get_embUrl()
        default_intent = normalizer.get_default_intent()
        linearsvc.generate_web2nl_svm_model(final_model,sigmoid_params,embUrl,workDirectory,default_intent,postProcessingRules)
        
        end_time = time.time()
        logger.info("Total execution time: %s", str(end_time - start_time))
        Task.updateStatus(workDirectory, "Model Created Successfully")
        # Testing the generated model with a sample embedding

        #####################################################
        # Commenting out the Sample Model Testing 

        # predict_data = [-0.013920364, 0.036257308, -0.0129521089, 0.0408232771, -0.028414825, -0.0414866656, 0.0593121424, -0.010834734, -0.0752527863, 0.00885635428, -0.0253722053, -0.0177697204, 0.0186599493, -0.0256391391, 0.0234380066, -0.0773016587, -0.0316566825, -0.0631263256, 0.0349719524, 0.0574531853, 0.023238996, -0.045436997, -0.0101281833, 0.0945090726, -0.00627863966, -0.0144318808, -0.0221464783, 0.05660237, -0.0274071321, -0.0155591732, -0.0200789385, 0.016018793, -0.0412942581, -0.0393139832, -0.000309920084, -0.060246475, 0.0100558847, -0.00466894498, 0.0632540956, 0.0271395948, -0.049167525, -0.0744382665, -0.0379752405, -0.00430319738, 0.00322760385, 0.0640057325, 0.0204735212, 0.0317621119, 0.0782235563, 0.000409479544, 0.102318235, 0.0111075761, -0.00210383092, 0.0269375201, -0.0981516913, -0.0987962708, -0.0397902802, 0.0466106646, -0.0146568827, -0.0616476759, -0.0138983596, -0.0188524649, -0.00971203949, 0.00786324, -0.0312367436, 0.018354889, 0.124083683, -0.0363393612, 0.0424477421, -0.0223345309, -0.00965169072, 0.0628261417, -0.0802845731, 0.0155033637, -0.00394708663, 0.0691119656, 0.0426003635, -0.0294282623, 0.0635628328, -0.0559182838, 0.0321351215, -0.0403533839, 0.0318166278, 0.00476710266, 0.0291559957, -0.0112757469, -0.0634697899, 0.0207079928, -0.0707175434, -0.052886039, 0.040358, 0.050995104, -0.0202901904, 0.014894709, 0.0139680775, 0.00660479115, -0.0115304086, 0.0365078971, -0.0125065017, 0.0349091925, -0.0131017072, -0.0524311811, 0.0666976348, -8.93958568e-05, 0.0471584573, -0.0160147, 0.0645075291, 0.0185540114, 0.0435852818, -0.04103994, -0.0226155818, -0.0435583219, 0.0381639302, -0.0305650756, -0.0292315092, -0.033290159, -6.18956765e-05, 0.0107742613, 0.00592195056, -0.0667529255, -0.0271448176, 0.0272689424, -0.0737589821, -0.0109681739, -0.0249249637, -0.0270688422, -0.00184530753, 0.0372405834, -0.0372263081, -0.05867441, -0.0360045657, 0.092005372, 0.0564016849, -0.015439732, -0.0366006531, 0.00829216745, -0.0204630941, -0.00554982852, -0.0316675268, 0.0137738874, 0.0348304547, -0.0709751621, -0.0381696597, 0.00991368387, -0.0147448955, 0.00393683231, -0.0671673, 0.017396884, 0.0723562539, 0.0215300359, 0.00971963, -0.0116287423, -0.0576360151, -0.0458955355, -0.0162969679, 0.107826941, 0.109651692, -0.0128303422, -0.0930878893, 0.0192907024, 0.069570072, 0.0769208893, -0.0113902651, 0.0190652106, -0.0501554757, -0.014719205, 0.0333291627, -0.105160527, 0.0133304475, -0.0449386574, 0.0209050961, 0.0690611601, -0.0543467477, 0.0333834812, 0.01785977, -0.0340317748, 0.035667479, 0.00819046516, 0.0379585065, 0.0328235701, -0.0256334413, -0.0276441425, 0.0172509272, -0.098068662, -0.0555647127, 0.0659801811, -0.0608881563, 0.0301388707, -0.0738570467, -0.0565897748, -0.0372921787, 0.0364776962, 0.0231389757, 0.0358704142, 0.0285687782, 0.00371384434, 0.0411620364, -0.00234244508, 0.0443589352, -0.00165628223, 0.0858624578, 0.00236681569, 0.0348137245, -0.0237678327, -0.0132989064, -0.0577970184, 0.0213098899, -0.0386476628, 0.0431055538, -0.0388227, 0.0243494697, 0.0523939617, -0.0383957848, -0.00325245899, -0.0150736384, 0.0187992975, 0.0158568937, -0.00742329797, -0.0459958501, -0.00470094709, 0.0436656624, 0.00414965441, -0.02690221, 0.029587185, -0.0175585076, -0.0485588945, 0.00325709069, -0.0120136682, -0.0321948305, -0.00140350335, -0.0123980027, 0.0026977146, -0.0621668771, -0.000841787609, 0.00395312067, 0.0132919727, -0.00745382905, -0.0139174601, 0.0161153357, -0.0379543491, 0.0187321901, -0.0138758374, -0.0299501605, -0.017428305, -0.0606935471, -0.0255726967, 0.050386, -0.0420826375, -0.00614841329, -0.0261052419, 0.013134921, 0.0643442, 0.110371761, 0.017015446, -0.030722633, -0.000390997535, 0.0862844139, -0.0216319114, 0.0368881263, -0.0140160099, 0.000260692876, 0.0159144104, 0.0360402204, -0.0434153676, 0.00246113469, 0.00953556225, -0.0370055959, -0.0872283205, -0.0431969, -0.0136289643, -0.0112739429, -0.0889231488, -0.0320453122, 0.0190046281, -0.0564102642, 0.00855684746, 0.0199958, 0.0601681471, -0.0353451818, 0.0535611436, 0.0873597637, 0.0170953255, -0.102777593, 0.0113341482, -0.0469827801, -0.0563701317, 0.00918948, -0.125241846, 0.0169365499, 0.0266107321, -0.00483281631, 0.0346935391, 0.0654789954, -0.0617779866, 0.0166538786, 0.044893831, -0.0242698286, 0.0867782086, 0.0179838892, -0.020050779, 0.0865681171, -0.00936979614, 0.00470853783, 0.0116428565, -0.0357693434, 0.0512959138, -0.018128721, 0.0280739758, -0.0359396935, -0.0119614778, 0.0744570866, -0.00948957074, 0.0209956989, 0.0637998283, -0.0730380788, -0.0430655442, -0.0220941, 0.0747142062, -0.0270399097, -0.102256671, 0.00416549388, -0.0233347863, 0.0421686098, -0.0583428368, -0.0349161327, 0.0364864, -0.0655263066, -0.0220748149, -0.0539502949, 0.0328205749, -0.00986089557, -0.0216328893, -0.0170260519, 0.044395335, -0.000519496039, 0.0616766475, 0.0272015762, 0.0393138342, 0.0102573885, -0.0610002503, 0.100392692, -0.0355404094, 0.00374236354, 0.0228859242, 0.0641628057, 0.0543300249, -0.0224269889, -0.0550110228, -0.00418475317, -0.0498525761, -0.042650979, -0.0111807473, -0.13964653, -0.0345630907, -0.05700076, -0.0333686545, 0.0164924469, -0.0374396741, 0.00795064215, -0.0366504528, 0.0440586321, 0.0380838923, 0.0287693404, -0.0262227636, -0.00073651562, -0.0257056467, 0.0582340956, 0.0282914247, 0.0215879176, 0.0144977886, 0.0140383104, -0.0195617955, -0.0357107446, 0.0831214935, -0.0123450793, 0.00735092442, -0.012169499, 0.0231842678, 0.0807605833, -0.0192973334, -0.0483252294, 0.0499785, -0.0508999974, -0.0126914233, -0.0664176494, 0.057802625, 0.00105102337, 0.0388422571, -0.0470065065, -0.0334214158, 0.0768226609, -0.0843447447, -0.0726598054, -0.0139268069, -0.0399020538, -0.0556353889, -0.0172302313, -0.0289492477, 0.0203803014, -0.023852637, -0.0396186151, 0.0199397486, 0.0283522923, 0.00794338528, -0.0315045863, -0.0397389643, 0.0765187293, -0.0242355932, 0.0176892281, -0.012406271, -0.0299667493, 0.0431958288, -0.0880135447, -0.0685217604, -0.00397710362, -0.0519419834, -0.0295789484, -0.00716308551, -0.0948615447, -0.00585699407, 0.0725328252, 0.0237517115, -0.0325622149, 0.010320791, -0.0320779458, 0.00762715237, 0.000270011224, 0.0858827308, 0.0615743063, -0.0526925, -0.0268527362, 0.0907346159, -0.0654151663, 0.00350541063, -0.0436422415, 0.0223961249, 0.0458065718, -0.109835945, 0.00406929711, -0.00608881656, 0.0438652262, -0.0942246094, 0.046423845, 0.0133042093, -0.0681899413, -0.0653481111, 0.0140457079, -0.0243151, -0.0208187681, -0.0153620429, 0.0561059527, 0.0319543667, 0.0292620026, -0.0549707748, -0.0262582451, -0.0428985171, -0.0400096402, 0.0580514446, 0.0366658382, 0.0419839099, 0.00734334253, -0.035254363, -0.0692199469, 0.0351634361, -0.0335196704, 0.0774920434, 0.0901381075, -0.010195137, -0.0253957417, 0.0530642718, -0.00700905686, -0.000255832245, -0.105427705, -0.0359593257, -0.0404231027, 0.0184417181, 0.0648131445, 0.00814706087, -0.0106319962, 0.0222731512, -0.00597881339, 0.0490439236, 0.0283836, -0.0201919079, -0.0244775359, 0.0134055177, 0.0170118455, 0.0641426593, -0.0100443447, 0.0674064532, 0.0115791494, -0.0235288497, -0.0187724028, -0.00822607614, -0.0166739598, 0.0391252898, 0.0205165818, 0.0873645544, -0.0227713138, -0.0528248139, -0.0146029936, -0.0671585798, -0.0563633256, -0.0101370718, -0.0569219925, 0.0193388574, 0.0109494096, 0.0569717437, 0.0300642978, -0.0318856351, -0.00490588136, 0.0243517626]
        # final_model_coef = final_model.coef_
        # dec_score_compute = []
        # for idx in range(len(final_model_coef)):
        #     score = 0
        #     for (feat,coef) in zip(predict_data,final_model_coef[idx]):
        #         score = score + feat*coef
        #     score = score + final_model.intercept_[idx]
        #     dec_score_compute.append(score)
        # res = final_model.decision_function([predict_data])
        # res_intent = final_model.predict([predict_data])
        # res_prob = []
        # for i in range(len(res[0])):
        #     dec_value = res[0][i]
        #     a = sigmoid_params[i][0]
        #     b = sigmoid_params[i][1]
        #     res_prob.append(linearsvc.sigmoid_predict(dec_value,a,b))
        # print("Prediction score from svm : " + str(res))
        # print("Probablity score after passing to sigmoid function : "+str(res_prob))
        # print("Intent with maximum probability " + str(y_pred_df.columns[res_prob.index(max(res_prob))]))
    except Exception as e:
        logger.exception("Model building failure for Tensorflow embedding: %s",str(e))
        Task.updateStatus(workDirectory, "Model building failure: " + str(e))
        Task.updateStatus(workDirectory, "Model building failure")

if __name__ == '__main__' :
    ModelLogger = gogo.Gogo(__name__, low_hdlr=gogo.handlers.file_hdlr(data.get("LOG_LOCATION")),
                            low_level='info',
                            low_formatter=formatter, high_formatter=formatter)
    Modellogger = ModelLogger.get_logger(modeluuid='rand_mid_1')
    normalization_uuid = 'rand_id_2'
    intent='None_None'
    tfx_normalization('rand_mid_1', Modellogger, intent,normalization_uuid,None,'use_large',None,'v3')
