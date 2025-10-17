import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from prompt_manager import build_prompt, build_prompt_first_word_prediction
from utils import (Log, log_system_info, hf_login,
                   find_token_length_distribution, TextDataset, get_emotion_logits, emotion_to_token_ids,
                   probe_classification, probe_classification_non_linear, extract_hidden_states, apply_zero_intervention_and_extract_logits, apply_random_intervention_and_extract_logits,
                   activation_patching, probe_regression, make_projections, promote_vec, seed_everywhere, apply_probes_on_hs)


##############################################################################################
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--model_index", type=int, default=0, help="Index of the model to use, 0: Llama3.2_1B, 1: Llama3.1_8B, 2: Gemma-2-2b-it, 3: Gemma-2-9b-it, 4: Phi-3.5-mini-instruct, 5: Phi-3-medium-128k-instruct, 6: OLMo-1B-hf, 7: OLMo-7B-0724-Instruct-hf, 8: Ministral-8B-Instruct-2410, 9: Mistral-Nemo-Instruct-2407, 10: OLMo-2-1124-7B-Instruct, 11: OLMo-2-1124-13B-Instruct")
parser.add_argument("--bs", type=int, default=4, help="Batch Size")

parser.add_argument("--prompt_type", type=str, default='joy_sadness_0')
parser.add_argument("--task_type", type=str, default='Emotion', choices=['Emotion', 'FirstWord'])

parser.add_argument("--open_vocab", action="store_true", default=True, help="Generate open vocab predictions")
parser.add_argument("--in_domain", action="store_true", default=True, help="Generate in-domain emotion labels and filter the dataset to only keep the samples with correctly predicted labels.")
parser.add_argument("--bbox_emotion_regression", action="store_true", default=True, help="Perform black-box emotion regression using appraisals, i.e. use appraisals as features to predict emotion labels")
parser.add_argument("--save_clean_logits", action="store_true", default=True, help="Save the original clean logits")

parser.add_argument("--extract_hidden_states", action="store_true", default=True, help="Extract hidden states")
parser.add_argument("--emotion_probing", action="store_true", default=True, help="Perform emotion probing")
parser.add_argument("--emotion_probing_non_linear", action="store_true", default=True, help="Perform emotion probing")
parser.add_argument("--appraisal_probing", action="store_true", default=True, help="Perform appraisal probing")

parser.add_argument("--extract_weights", action="store_true", default=True, help="Extract weights from probes")
parser.add_argument("--zero_intervention", action="store_true", default=True, help="Perform zero intervention")
parser.add_argument("--random_intervention", action="store_true", default=True, help="Perform random intervention")
parser.add_argument("--activation_patching", action="store_true", default=True, help="Perform activation patching")
parser.add_argument("--attention_weights", action="store_true", default=True, help="Extract attention weights")

parser.add_argument("--emotion_promotion", action="store_true", default=True, help="Perform emotion promotion")
parser.add_argument("--appraisal_surgery", action="store_true", default=False, help="Perform appraisal surgery")

args = parser.parse_args()
##############################################################################################


log = Log(log_name='intervention')
logger = log.logger
log_system_info(logger)
hf_login(logger)
BATCH_SIZE = args.bs
HOOKED = True
device_map = 'cuda'

os.makedirs('outputs/', exist_ok=True)

train_data = pd.read_csv('data/enVent_gen_Data.csv', encoding='ISO-8859-1')
train_data['emotion'] = train_data['emotion'].replace('no-emotion', 'neutral')

appraisals = ['predict_event', 'pleasantness', 'other_responsblt', 'chance_control', 'suddenness', 'familiarity', 'unpleasantness', 'goal_relevance', 'self_responsblt', 'predict_conseq', 'goal_support', 'urgency', 'self_control', 'other_control', 'accept_conseq', 'standards', 'social_norms', 'attention', 'not_consider', 'effort']
            #   , 'extravert', 'critical', 'dependable', 'anxious', 'open', 'quiet', 'sympathetic',  'calm', 'conventional']
assert len(appraisals) == len(set(appraisals)), "Duplicate appraisals found in the list"
appraisals_to_id = {appraisal: i for i, appraisal in enumerate(appraisals)}


if args.task_type == 'FirstWord':
    first_words_list = ['I', 'My', 'my', 'When', 'when', 'i', 'A', 'a', 'The', 'the', 'someone', 'Someone', 'we']
    first_word_to_id = {word: i for i, word in enumerate(first_words_list)}
    id_to_first_word = {v: k for k, v in first_word_to_id.items()}
    emotions_list, emotion_to_id, id_to_emotion = first_words_list, first_word_to_id, id_to_first_word # for compatibility with the rest of the code
    
    # filter out the rows that start with this first word
    train_data['emotion_id'] = train_data['hidden_emo_text'].apply(lambda x: first_word_to_id[x.split()[0]] if x.split()[0] in first_words_list else -1)
    train_data = train_data[train_data['emotion_id'] != -1]
    
    func = build_prompt_first_word_prediction()
    train_data['emotion'] = train_data['hidden_emo_text'].apply(lambda x: x.split()[0])
    
    
elif args.task_type == 'Emotion':
    emotions_list = ['anger', 'boredom', 'disgust', 'fear', 'guilt', 'joy', 'neutral', 'pride', 'relief', 'sadness', 'shame', 'surprise', 'trust']
    emotion_to_id = {emotion: i for i, emotion in enumerate(emotions_list)}
    id_to_emotion = {v: k for k, v in emotion_to_id.items()}
    train_data['emotion_id'] = train_data['emotion'].map(emotion_to_id).astype(int)

    if '_' in args.prompt_type:
        shots = args.prompt_type.split('_')[:-1]
        prompt_index = int(args.prompt_type.split('_')[-1])
    else:
        shots = []
        prompt_index = int(args.prompt_type)

    func = build_prompt(shots=shots, prompt_index=prompt_index)

train_data['input_text'] = train_data['hidden_emo_text'].apply(func)

labels = torch.from_numpy(train_data[['emotion_id'] + appraisals].to_numpy())

dataset = TextDataset(train_data['input_text'].tolist(), labels)
dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=False)

from LLMs.my_llama import LlamaForCausalLM
from LLMs.my_phi3 import Phi3ForCausalLM
from LLMs.my_gemma2 import Gemma2ForCausalLM
from LLMs.my_olmo import OlmoForCausalLM
from LLMs.my_mistral import MistralForCausalLM
from LLMs.my_olmo2 import Olmo2ForCausalLM
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

model_names =       ['meta-llama/Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct', 
                     'google/gemma-2-2b', 'google/gemma-2-9b-it', 
                     'microsoft/Phi-3.5-mini-instruct', 'microsoft/Phi-3-medium-128k-instruct', 
                     'allenai/OLMo-1B-hf', 'allenai/OLMo-7B-0724-Instruct-hf', 
                     'mistralai/Ministral-8B-Instruct-2410', 'mistralai/Mistral-Nemo-Instruct-2407',
                     'allenai/OLMo-2-1124-7B-Instruct', 'allenai/OLMo-2-1124-13B-Instruct'
                     ]

model_short_names = ['Llama3.2_1B', 'Llama3.1_8B',
                     'gemma-2-2b', 'gemma-2-9b-it',
                     'Phi-3.5-mini-instruct', 'Phi-3-medium-128k-instruct',
                     'OLMo-1B-hf', 'OLMo-7B-0724-Instruct-hf',
                     'Ministral-8B-Instruct-2410', 'Mistral-Nemo-Instruct-2407',
                     'OLMo-2-1124-7B-Instruct', 'OLMo-2-1124-13B-Instruct']

model_classes =     [LlamaForCausalLM, LlamaForCausalLM,
                     Gemma2ForCausalLM, Gemma2ForCausalLM,
                     Phi3ForCausalLM, Phi3ForCausalLM,
                     OlmoForCausalLM, OlmoForCausalLM,
                     MistralForCausalLM, MistralForCausalLM,
                     Olmo2ForCausalLM, Olmo2ForCausalLM]

model_name, model_short_name, model_class = list(zip(model_names, model_short_names, model_classes))[args.model_index]

save_prefix = ''
if not HOOKED:
    logger.info("Using an unhooked model ...")
    model_class = AutoModelForCausalLM
    save_prefix = 'UNHOOKED_'

logger.info(f"Model Name: {model_name}")

if args.prompt_type == 'joy_sadness_0': # default mode, no need to change path_prefix
    args.path_prefix = ''
else:
    args.path_prefix = args.prompt_type + '_'

if args.task_type == 'FirstWord':
    args.path_suffix = '_FirstWord'
else:
    args.path_suffix = ''

model_short_name =  args.path_prefix + model_short_name + args.path_suffix

os.makedirs(f'outputs/{model_short_name}', exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    padding_side='left',
    cache_dir="/shared/3/projects/bangzhao/models/transformers",
    local_files_only=False, 
)

if tokenizer.pad_token is None:
    logger.info("Adding padding token to the tokenizer")
    tokenizer.pad_token = tokenizer.eos_token

model = model_class.from_pretrained(
    model_name,
    device_map=device_map,
    cache_dir="/shared/3/projects/bangzhao/models/transformers", 
    local_files_only=False,
)

token_length_distribution = find_token_length_distribution(train_data['input_text'], tokenizer)
logger.info(f"Token Length Distribution: {token_length_distribution}")
assert token_length_distribution['max_length'] <= tokenizer.model_max_length, "Token length exceeds model's max length."
logger.info(f"Original Model_Max_Length: {tokenizer.model_max_length}")

num_params = sum(p.numel() for p in model.parameters())
size_on_memory = sum(p.numel() * p.element_size() for p in model.parameters())
logger.info(f"Loaded model '{model_name}' with {num_params}  parameters ({size_on_memory / (1024 ** 2):.2f} MB)")
logger.info(f"Model configuration: {model.config}")

emotions_to_tokenized_ids = emotion_to_token_ids(emotions_list, tokenizer)


if args.open_vocab:
    logger.info("------------------------------ Generating Open Vocab Predictions ------------------------------")
    open_vocab_preds = get_emotion_logits(dataloader, tokenizer, model, ids_to_pick=None, apply_argmax=True)
    preds = tokenizer.batch_decode(open_vocab_preds, skip_special_tokens=True)
    gt = [' ' + id_to_emotion[i.item()] for i in labels[:, 0]]
    
    torch.save([gt, preds], f'outputs/{model_short_name}/{save_prefix}open_vocab_predictions.pt')
    

if args.in_domain:
    logger.info("------------------------------ Generating In-Domain Emotion Labels using Logits ------------------------------")
    probs = get_emotion_logits(dataloader, tokenizer, model, ids_to_pick=emotions_to_tokenized_ids)
    
    torch.save([probs.argmax(dim=1), labels[:, 0]], f'outputs/{model_short_name}/{save_prefix}unfiltered_emotion_logits.pt')
    
    emotion_labels = [id_to_emotion[i.item()] for i in probs.argmax(dim=1)]
    filtered_train_data = train_data[train_data['emotion'] == emotion_labels]
    logger.info(f"Total mismatched labels: { len(train_data) - len(filtered_train_data)}/{len(train_data)} ({ 1 - len(filtered_train_data) / len(train_data):.2%})")
    filtered_train_data.to_csv(f'outputs/{model_short_name}/filtered_train_data.csv', index=False)

assert HOOKED, "Unhooked models are not supported for the following experiments"


try:
    train_data = pd.read_csv(f'outputs/{model_short_name}/filtered_train_data.csv')
    labels = torch.from_numpy(train_data[['emotion_id'] + appraisals].to_numpy())
except:
    raise Exception("Filtered train data not found, run the code again with --in_domain enabled")

dataset = TextDataset(train_data['input_text'].tolist(), labels)
dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=False)

if args.bbox_emotion_regression:
    logger.info("------------------------------ Black-box appraisal emotion regression ------------------------------")

    reg_input = labels[:, 1:]
    reg_labels = labels[:, 0]
    r = probe_classification(reg_input, reg_labels, return_weights=True)
    w = r['weights']
    b = r['bias']
    
    print(f'Emotion Prediction Acc: train {r["accuracy_train"]}, test {r["accuracy_test"]}')

    torch.save({'emotion_to_id': emotion_to_id, 'appraisals_to_id': appraisals_to_id, 
                'labels': labels, 'weights': w, 'bias': b}, f'outputs/{model_short_name}/emotion_appraisal_labels.pt')
    
if args.save_clean_logits:
    logger.info("------------------------------ Saving Original Clean Logits ------------------------------")
    original_logits = get_emotion_logits(dataloader, tokenizer, model, ids_to_pick=emotions_to_tokenized_ids)
    torch.save(original_logits, f'outputs/{model_short_name}/original_logits.pt')

if args.extract_hidden_states:
    logger.info("------------------------------ Extracting Hidden States ------------------------------")
    extraction_locs = [3, 6, 7]
    extraction_tokens = [-1]#, -2, -3, -4, -5]
    extraction_layers = list(range(model.config.num_hidden_layers))

    all_hidden_states = extract_hidden_states(dataloader, tokenizer, model, logger, extraction_locs=extraction_locs, extraction_layers=extraction_layers, extraction_tokens = extraction_tokens)
    # torch.save(all_hidden_states, f'outputs/{model_short_name}/hidden_states_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt')

if args.emotion_probing:
    logger.info("--------------------------------- Emotion Probing ---------------------------------")

    extraction_locs = [3, 6, 7]
    extraction_tokens = [-1]#, -2, -3, -4, -5]
    extraction_layers = list(range(model.config.num_hidden_layers))

    if not args.extract_hidden_states:
        try:
            all_hidden_states = torch.load(f'outputs/{model_short_name}/hidden_states_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt', weights_only=False)
        except:
            raise Exception("Hidden states not found, run the code again with --extract_hidden_states")
        
    
    size_on_memory = all_hidden_states.element_size() * all_hidden_states.numel()
    logger.info(f"Hidden states tensor size: {size_on_memory / (1024 ** 2):.2f} MB")

    results = {}
    for i, layer in tqdm(enumerate (extraction_layers), total=len(extraction_layers)):
        results[layer] = {}
        for j, loc in enumerate (extraction_locs):
            results[layer][loc] = {}
            for k, token in enumerate (extraction_tokens):
                results[layer][loc][token] = probe_classification(all_hidden_states[:, i, j, k], labels[:, 0], return_weights=True)
                

    torch.save(results, f'outputs/{model_short_name}/emotion_probing_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt')

if args.emotion_probing_non_linear:
    logger.info("--------------------------------- Emotion Non-Linear Probing ---------------------------------")

    extraction_locs = [3, 6, 7]
    extraction_tokens = [-1]#, -2, -3, -4, -5]
    extraction_layers = list(range(model.config.num_hidden_layers))

    if not args.extract_hidden_states:
        try:
            all_hidden_states = torch.load(f'outputs/{model_short_name}/hidden_states_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt', weights_only=False)
        except:
            raise Exception("Hidden states not found, run the code again with --extract_hidden_states")
    
    size_on_memory = all_hidden_states.element_size() * all_hidden_states.numel()
    logger.info(f"Hidden states tensor size: {size_on_memory / (1024 ** 2):.2f} MB")

    results = {}
    for i, layer in tqdm(enumerate (extraction_layers), total=len(extraction_layers)):
        results[layer] = {}
        for j, loc in enumerate (extraction_locs):
            results[layer][loc] = {}
            for k, token in enumerate (extraction_tokens):
                results[layer][loc][token] = probe_classification_non_linear(all_hidden_states[:, i, j, k], labels[:, 0], return_weights=True)
                

    torch.save(results, f'outputs/{model_short_name}/non_linaer_emotion_probing_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt')    

if args.appraisal_probing:

    logger.info("------------------------------ Appraisal Probing ------------------------------")

    results = {}

    extraction_locs = [3, 6, 7]
    extraction_tokens = [-1]#, -2, -3, -4, -5]
    extraction_layers = list(range(model.config.num_hidden_layers))

    if not args.extract_hidden_states:
            try:
                all_hidden_states = torch.load(f'outputs/{model_short_name}/hidden_states_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt', weights_only=False)
            except:
                raise Exception("Hidden states not found, run the code again with --extract_hidden_states")
    

    for app in tqdm(appraisals, total=len(appraisals)):
        results[app] = {}
        for i, layer in tqdm(enumerate(extraction_layers), total=len(extraction_layers)):
            results[app][layer] = {}
            for j, loc in enumerate(extraction_locs):
                results[app][layer][loc] = {}
                for k, token in enumerate(extraction_tokens):
                    results[app][layer][loc][token] = probe_regression(all_hidden_states[:, i, j, k],
                                                                       labels[:, appraisals_to_id[app] + 1],
                                                                       return_weights=True)

    torch.save(results,
               f'outputs/{model_short_name}/appraisal_probing_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt')

if args.extract_weights:
    logger.info("------------------------------ Extracting Weights From Probes ------------------------------")
    extraction_locs = [3, 6, 7]
    extraction_tokens = [-1]#, -2, -3, -4, -5]
    extraction_layers = list(range(model.config.num_hidden_layers))

    try:
        emotion_probing_results = torch.load(f'outputs/{model_short_name}/emotion_probing_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt', weights_only=False)
    except:
        raise Exception("Emotion probing results not found, run the code again with --emotion_probing enabled")
    

    emotions_weights_ = torch.zeros([len(emotions_list), len(extraction_layers), len(extraction_locs), len(extraction_tokens), model.config.hidden_size])
    emotions_biases_ = torch.zeros([len(emotions_list), len(extraction_layers), len(extraction_locs), len(extraction_tokens)])
    
    for j, layer in enumerate(extraction_layers):
        for k, loc in enumerate(extraction_locs):
            for l, token in enumerate(extraction_tokens):
                emotions_weights_[:, j, k, l, :] = torch.from_numpy(emotion_probing_results[layer][loc][token]['weights'])
                emotions_biases_[:, j, k, l] = torch.from_numpy(emotion_probing_results[layer][loc][token]['bias'])

    torch.save(emotions_weights_, f'outputs/{model_short_name}/emotions_weights_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt')

    torch.save(emotions_biases_, f'outputs/{model_short_name}/emotions_biases_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt')
    
    try:
        appraisal_probing_results = torch.load(f'outputs/{model_short_name}/appraisal_probing_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt', weights_only=False)
    except:
        raise Exception("Appraisal probing results not found, run the code again with --appraisal_probing enabled")

    appraisals_weights = torch.zeros([len(appraisals), len(extraction_layers), len(extraction_locs), len(extraction_tokens), model.config.hidden_size])
    appraisals_biases = torch.zeros([len(appraisals), len(extraction_layers), len(extraction_locs), len(extraction_tokens)])

    for i, app in enumerate(appraisals):
        for j, layer in enumerate(extraction_layers):
            for k, loc in enumerate(extraction_locs):
                for l, token in enumerate(extraction_tokens):
                    appraisals_weights[i, j, k, l, :] = torch.from_numpy(appraisal_probing_results[app][layer][loc][token]['weights'][0])
                    appraisals_biases[i, j, k, l] = torch.from_numpy(appraisal_probing_results[app][layer][loc][token]['bias'])

    torch.save(appraisals_weights, f'outputs/{model_short_name}/appraisals_weights_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt')
    torch.save(appraisals_biases, f'outputs/{model_short_name}/appraisals_biases_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt')

if args.zero_intervention:
    logger.info("------------------------------ Performing Zero Intervention ------------------------------")

    span = 3
    intervention_layers = list(range(span // 2, model.config.num_hidden_layers - span // 2))
    intervention_locs = [[3], [6], [7]]

    for intervention_tokens in [[-1]]: #[-2], [-3], [-4], [-5], 'all'
        for loc in intervention_locs:
            intervention_results = {}
            for center_layer in tqdm(intervention_layers, total=len(intervention_layers)):
                layers = list(range(center_layer - span // 2, center_layer + span // 2 + 1))
                intervention_results[center_layer] = apply_zero_intervention_and_extract_logits(dataloader, tokenizer, model, logger,
                                                                            intervention_layers=layers, intervention_locs=loc,
                                                                            intervention_tokens=intervention_tokens, ids_to_pick=emotions_to_tokenized_ids)

            torch.save(intervention_results, f'outputs/{model_short_name}/zero_intervention_results_layers_{intervention_layers}_span_{span}_locs_{loc}_token_{intervention_tokens}.pt')

if args.random_intervention:
    logger.info("------------------------------ Performing Random Intervention ------------------------------")

    span = 3
    
    seed = 100
    seed_everywhere(seed)    
    
    intervention_layers = list(range(span // 2, model.config.num_hidden_layers - span // 2))
    intervention_locs = [[3], [6], [7]]
    

    for intervention_tokens in [[-1]]: #[-2], [-3], [-4], [-5], 'all'
        for loc in intervention_locs:
            intervention_results = {}
            for center_layer in tqdm(intervention_layers, total=len(intervention_layers)):
                layers = list(range(center_layer - span // 2, center_layer + span // 2 + 1))
                intervention_results[center_layer] = apply_random_intervention_and_extract_logits(dataloader, tokenizer, model, logger,
                                                                            intervention_layers=layers, intervention_locs=loc,
                                                                            intervention_tokens=intervention_tokens, ids_to_pick=emotions_to_tokenized_ids)

            torch.save(intervention_results, f'outputs/{model_short_name}/random_intervention_results_layers_{intervention_layers}_span_{span}_locs_{loc}_token_{intervention_tokens}_seed_{seed}.pt')

if args.activation_patching:
    logger.info("------------------------------ Performing Activation Patching ------------------------------")
    num_experiments = 200
    span = 5
    layers_ = list(range(span // 2, model.config.num_hidden_layers - span // 2))

    for locs in [[3], [6], [7]]:
        for intervention_tokens in [[-1]]: #, [-2], [-3], [-4], [-5]
            patching_results = {}
            for center_layer in tqdm(layers_, total=len(layers_)):
                layers = list(range(center_layer - span // 2, center_layer + span // 2 + 1))

                patching_results[center_layer] = {}
                for i in range(num_experiments):
                    # sample a random source sentence and target sentence with different emotions
                    source_sentence = train_data.sample(1)
                    target_sentence = train_data.sample(1)
                    while source_sentence['emotion'].values[0] == target_sentence['emotion'].values[0]:
                        target_sentence = train_data.sample(1)

                    source_sentence, source_emotion = source_sentence['input_text'].values[0], source_sentence['emotion'].values[0]
                    target_sentence, target_emotion = target_sentence['input_text'].values[0], target_sentence['emotion'].values[0]

                    patching_results[center_layer][i] = activation_patching(source_sentence, target_sentence, tokenizer, model, logger, intervention_layers=layers, intervention_locs=locs, ids_to_pick=emotions_to_tokenized_ids, intervention_tokens=intervention_tokens)


            torch.save(patching_results, f'outputs/{model_short_name}/patching_results_layers_{layers_}_locs_{locs}_span_{span}_token_{intervention_tokens}.pt')

if args.attention_weights:
    logger.info("------------------------------ Extracting Attention Weights ------------------------------")

    # create a new dataloader with batch size 1
    dataloader_1bs = DataLoader(dataset, batch_size = 1, shuffle=False)
    extraction_layers = list(range(model.config.num_hidden_layers))
    extraction_locs = [10]
    extraction_tokens = [-1]
    results = extract_hidden_states(dataloader_1bs, tokenizer, model, logger, extraction_locs=extraction_locs, extraction_layers=extraction_layers, extraction_tokens = extraction_tokens, do_final_cat = False, return_tokenized_input=True)
    torch.save(results, f'outputs/{model_short_name}/attention_weights_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt')

if args.clean_probings or args.emotion_promotion or args.appraisal_emotion_promotion or args.appraisal_surgery or args.appraisal_coeff_emotion_promotion or args.random_promotion or args.random_coeff:
    logger.info("------------------------------ Loading Weights ------------------------------")
    # Loading emotions weights
    extraction_locs = [3, 6, 7]
    extraction_tokens = [-1]#, -2, -3, -4, -5]
    extraction_layers = list(range(model.config.num_hidden_layers))
    emotions_weights_ = torch.load(f'outputs/{model_short_name}/emotions_weights_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt', weights_only=False)
    emotions_biases_ = torch.load(f'outputs/{model_short_name}/emotions_biases_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt', weights_only=False)
    
    # Loading appraisals weights
    appraisals_weights_ = torch.load(f'outputs/{model_short_name}/appraisals_weights_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt', weights_only=False)
    appraisals_biases_ = torch.load(f'outputs/{model_short_name}/appraisals_biases_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt', weights_only=False)

if args.emotion_promotion:
    logger.info("------------------------------ Emotion Promotion ------------------------------")
    os.makedirs(f'outputs/{model_short_name}/emotion_promotion/', exist_ok=True)
    
    promotion_tokens = [-1] #'all' #
    promotion_locs = [[7]] #, [3], [6]
    span = 3
    layers_ = list(range(span // 2, model.config.num_hidden_layers - span // 2))
    
    Beta1 = 1.0
    Beta2 = 0.0
    do_normalization = False
    
    emotions_weights = emotions_weights_[:, :, :, [-1] if promotion_tokens == 'all' else [extraction_tokens.index(token) for token in promotion_tokens]]
    
    for emotion_to_promote in ['anger', 'joy', 'pride', 'fear', 'guilt', 'sadness']:
        for locs in promotion_locs:
            print(f"================= Promoting emotion: {emotion_to_promote} at loc: {locs}, Beta1: {Beta1}, Beta2: {Beta2}, Normalization: {do_normalization}")
            
            results = {}
            results_probe = {}
            for center_layer in tqdm(layers_, total=len(layers_)):
                layers_with_span = list(range(center_layer - span // 2, center_layer + span // 2 + 1))
                
                ews = emotions_weights[:, [extraction_layers.index(layer) for layer in layers_with_span]][:, :, [extraction_locs.index(loc) for loc in locs]]
                
                projection = make_projections(ews.permute(1, 2, 3, 0, 4)) # [layers, locs, tokens, hidden_size, hidden_size]
                ew = ews[emotions_list.index(emotion_to_promote)]
                
                if do_normalization:
                    ew = ew / torch.norm(ew, dim=-1, keepdim=True)
                
                ew = Beta1 * ew
                
                results[center_layer], _ = promote_vec(dataloader, tokenizer, model, logger, prom_vector = ew, projection_matrix = projection, Beta = Beta2,
                                                                                promotion_layers = layers_with_span, promotion_locs = locs, promotion_tokens = promotion_tokens, 
                                                                                ids_to_pick = emotions_to_tokenized_ids)
                
            torch.save(results,       f'outputs/{model_short_name}/emotion_promotion/emotion_promotion_{emotion_to_promote}_layers_{layers_}_span_{span}_locs_{locs}_tokens_{promotion_tokens}_Beta1_{Beta1}_Beta2_{Beta2}_normalization_{do_normalization}.pt')

if args.appraisal_surgery:
    logger.info("------------------------------ Appraisal Surgery ------------------------------")
    os.makedirs(f'outputs/{model_short_name}/appraisal_surgery/', exist_ok=True)
    
    surgery_appraisals = [
                          (['pleasantness'], [], [+1]),
                          (['pleasantness'], [], [-1]),
                          
                          (['other_responsblt'], [], [+1]),
                          (['other_responsblt'], [], [-1]),
                          
                          (['pleasantness'], ['predict_event', 'other_responsblt', 'chance_control'], [+1]),
                          (['other_responsblt'], ['predict_event', 'pleasantness', 'chance_control'], [+1]),
                          (['predict_event'], ['pleasantness', 'other_responsblt', 'chance_control'], [+1]),
                          (['chance_control'], ['pleasantness', 'other_responsblt', 'predict_event'], [+1]),

                          (['pleasantness'], ['predict_event', 'other_responsblt', 'chance_control'], [-1]),
                          (['other_responsblt'], ['predict_event', 'pleasantness', 'chance_control'], [-1]),
                          (['predict_event'], ['pleasantness', 'other_responsblt', 'chance_control'], [-1]),
                          (['chance_control'], ['pleasantness', 'other_responsblt', 'predict_event'], [-1]),
                          
                          (['pleasantness', 'other_responsblt'], ['predict_event', 'chance_control'], [+1, -1]),
                          (['pleasantness', 'other_responsblt'], ['predict_event', 'chance_control'], [-1, -1]),
                          
                          (['pleasantness', 'other_responsblt'], ['predict_event', 'chance_control'], [+1, +1]),
                          (['pleasantness', 'other_responsblt'], ['predict_event', 'chance_control'], [-1, +1]),
                          
                          (['pleasantness', 'predict_event'], ['other_responsblt', 'chance_control'], [+1, -1]),
                          (['pleasantness', 'predict_event'], ['other_responsblt', 'chance_control'], [-1, -1]),
                          
                          (['pleasantness', 'predict_event'], ['other_responsblt', 'chance_control'], [+1, +1]),
                          (['pleasantness', 'predict_event'], ['other_responsblt', 'chance_control'], [-1, +1]),                          
                          
                          (['random_42'], [], [+1]),
                          (['random_43'], [], [+1]),
                          (['random_44'], [], [+1]),
                          ]
    
    
    promotion_tokens = [-1]
    promotion_locs = [[7]] #, [3], [6]
    Beta1 = 1.0
    Beta2 = 0.0
    do_normalization = True
    
    span = 3
    layers_ = list(range(span // 2, model.config.num_hidden_layers - span // 2))

    appraisals_weights = appraisals_weights_[:, :, :, [-1] if promotion_tokens == 'all' else [extraction_tokens.index(token) for token in promotion_tokens]] # [appraisals, layers, locs, tokens, hidden_size]

    for appraisals_to_change_, appraisals_to_fix_, coeffs_ in surgery_appraisals:
        logger.info(f"================= Changing appraisal: {appraisals_to_change_} ({coeffs_}) loc {promotion_locs} with Beta1: {Beta1}, Beta2: {Beta2}, Normalization: {do_normalization}, while keeping {appraisals_to_fix_} fixed")
        
        assert len(appraisals_to_change_) > 0, "At least one appraisal should be changed"
        assert set(appraisals_to_change_).isdisjoint(set(appraisals_to_fix_)), "Appraisals to change and fix should be disjoint"
        
        for locs in promotion_locs:
            results = {}
            results_probe = {}
            for center_layer in tqdm(layers_, total=len(layers_)):
                layers_with_span = list(range(center_layer - span // 2, center_layer + span // 2 + 1))
                aws = appraisals_weights[:, [extraction_layers.index(layer) for layer in layers_with_span]][:, :, [extraction_locs.index(loc) for loc in locs]]
                
                
                changing_appraisals_weights = []
                for _, appraisals_to_change in enumerate(appraisals_to_change_):
                    if 'random' in appraisals_to_change:
                        seed = int(appraisals_to_change.split('_')[-1])
                        seed_everywhere(seed)
                        appraisals_to_change_weight = torch.randn_like(aws[0]) * aws[0].norm(dim = -1, keepdim = True)
                        changing_appraisals_weights.append(appraisals_to_change_weight)
                    else:
                        appraisals_to_change_index = appraisals_to_id[appraisals_to_change]
                        appraisals_to_change_weight = aws[appraisals_to_change_index]
                        changing_appraisals_weights.append(appraisals_to_change_weight)
                
                changing_appraisals_weights = torch.stack(changing_appraisals_weights, dim=0)
                changing_appraisals = changing_appraisals_weights.permute(1, 2, 3, 0, 4) # [layers, locs, tokens, appraisals, hidden_size]
                
                
                fixed_appraisals_weights = []
                for _, appraisals_to_fix in enumerate(appraisals_to_fix_):
                    if 'random' in appraisals_to_fix:
                        seed = int(appraisals_to_fix.split('_')[-1])
                        seed_everywhere(seed)
                        appraisals_to_fix_weight = torch.randn_like(aws[0]) * aws[0].norm(dim = -1, keepdim = True)
                        fixed_appraisals_weights.append(appraisals_to_fix_weight)
                    else:
                        appraisals_to_fix_index = appraisals_to_id[appraisals_to_fix]
                        appraisals_to_fix_weight = aws[appraisals_to_fix_index]
                        fixed_appraisals_weights.append(appraisals_to_fix_weight)
                    
                
                if len(fixed_appraisals_weights) > 0:
                    fixed_appraisals_weights = torch.stack(fixed_appraisals_weights, dim=0)
                    fixed_appraisals = fixed_appraisals_weights.permute(1, 2, 3, 0, 4) # [layers, locs, tokens, appraisals, hidden_size]
                    P_fixed = make_projections(fixed_appraisals) # [layers, locs, tokens, hidden_size, hidden_size]
                else:
                    P_fixed = torch.zeros([len(layers_with_span), len(locs), len(promotion_tokens), model.config.hidden_size, model.config.hidden_size])
                
                identity_proj = torch.eye(P_fixed.size(-1)).unsqueeze(0).unsqueeze(0).unsqueeze(0).broadcast_to(P_fixed.size())
                
                prom_vectors = torch.einsum('ijklm,ijkam->ijkal', (identity_proj - P_fixed), changing_appraisals)
                coeffs = torch.tensor(coeffs_).to(torch.float32)
                if len(coeffs.shape) == 0:
                    coeffs = coeffs.unsqueeze(0)
                prom_vector = torch.einsum('ijkal, a -> ijkl', changing_appraisals, coeffs)
                
                if do_normalization:
                    prom_vector = prom_vector / torch.norm(prom_vector, dim=-1, keepdim=True)
                
                prom_vector = Beta1 * prom_vector
                
                results[center_layer], hs = promote_vec(dataloader, tokenizer, model, logger, prom_vector = prom_vector, projection_matrix = identity_proj, Beta = Beta2,
                                                                                promotion_layers = layers_with_span, promotion_locs = locs, promotion_tokens = promotion_tokens, 
                                                                                ids_to_pick = emotions_to_tokenized_ids)

            appraisals_to_change = '_'.join(appraisals_to_change_)
            appraisals_to_fix = '_'.join(appraisals_to_fix_)
            coeffs = '_'.join([str(c) for c in coeffs_])
            
            torch.save(results, f'outputs/{model_short_name}/appraisal_surgery/appraisal_surgery_{appraisals_to_change}_fixed_{appraisals_to_fix}_coeffs_{coeffs}_layers_{layers_}_span_{span}_locs_{locs}_tokens_{promotion_tokens}_Beta1_{Beta1}_Beta2_{Beta2}_normalization_{do_normalization}.pt')