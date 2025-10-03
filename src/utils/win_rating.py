import numpy as np

from src.model.make_prediction import generate_sample

win_query = """For the following query to a chatbot, which response is more helpful?
FIRST provide a one-sentence comparison of the two responses and explain \
which you feel is more helpful. SECOND, on a new line, state only "A" or \
"B" to indicate which response is more helpful.\
Your response should use the format:
Comparison: <one-sentence comparison and explanation>
More helpful: <"A" or "B">

Query:
<Q>
Response A:
<RA>
Response B:
<RB>"""


def get_best_model(response):
    best_model = response
    best_model = best_model.split("More helpful:")[-1].strip()
    if "A" in best_model and "B" in best_model:
        print("Both models were selected")
        return 0.5
    elif "A" in best_model:
        return 0
    elif "B" in best_model:
        return 1
    return -0.5

def compete_dpo(test_questions, tokenizer, llm_evaluator, dpo_model, device, base_model=None, correct_answers=None, compete_answers=True):
    base_wins = 0
    dpo_wins = 0
    wins_history = []
    # Check existence of for what to compare
    if base_model is None and correct_answers is None:
        print("No comparison was provided")
        return None
    # Select another approach is available
    elif not compete_answers and base_model is None:
         compete_answers = True
    elif compete_answers and correct_answers is None:
         compete_answers = False

    for ind in range(len(test_questions)):
        sample = test_questions[ind]
        # Select answer to compare with dpo
        if compete_answers:
            base_answer = correct_answers[ind]
        else:
            base_answer = generate_sample(base_model, tokenizer, sample, device)
        dpo_answer = generate_sample(dpo_model, tokenizer, sample, device)

        straight_order = True
        queries = [base_answer, dpo_answer]
        # Change order randomly
        if np.random.choice([True, False]):
            straight_order = False
            queries = [dpo_answer, base_answer]
        print("Response A:")
        print(queries[0])
        print("Response B:")
        print(queries[1])
        # Ask for the best answer
        ask = win_query.replace("<Q>", sample).replace("<RA>", queries[0]).replace("<RB>", queries[1])
        resp = llm_evaluator(ask)
        win_num = get_best_model(resp)
        # Select a winner
        if win_num == 0.5:
            base_wins += 1
            dpo_wins += 1
        elif ((win_num == 0) and straight_order) or ((win_num == 1) and not straight_order):
            base_wins += 1
        elif ((win_num == 1) and straight_order) or ((win_num == 0) and not straight_order):
            dpo_wins += 1
        wins_history.append([sample, base_answer, dpo_answer, straight_order, ask, resp, win_num])

    return base_wins, dpo_wins, wins_history



