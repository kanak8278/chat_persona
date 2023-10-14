from nubia_score import Nubia
from tqdm import tqdm
import pandas as pd
n = Nubia()
paths = ["/work/kanakr/chat_persona/generator/bart_train/predictions/focus_inference_bart_base_20_LM.csv"]


for path in paths:
    print("================================================================================================")
    print(path)
    df = pd.read_csv(path)
    parent_path = path[:-4]
    print(df.columns)
    results = []
    exit()
    df = df.sample(frac=0.1, replace=True, random_state=1)
    print(df.shape)
    
    for idx, (dialogID, utterance, pred, ans) in tqdm(enumerate(zip(df['dialogID'], df['utterance'],df['prediction'], df['answer']))):
        result = {"dialogID":dialogID, "utterance":utterance}
        
        try:
            score = n.score(pred, ans, get_features=True)
            result['nubia_score'] = score['nubia_score']
            result.update(score['features'])
            results.append(result)
        except Exception as e:
            print(f"Index {idx}, {dialogID, utterance} has error!")
            print(e)
            print()
        if idx % 100:
            print(f"Index {idx} completed")
            break
    result_df = pd.DataFrame(results)
    nubia_s = list(result_df['nubia_score'])
    print("Nubia Score:", sum(nubia_s)/len(nubia_s))
    result_df.to_csv(f"{parent_path}_nubia_score_temp.csv", index=False)
    print()
