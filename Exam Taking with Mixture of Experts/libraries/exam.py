from .training import repeat_data

import json
import numpy as np
import os
import pandas as pd
import random
import subprocess
import torch

def write_exam_with_choices():
    data_test = pd.read_csv("data/polynomial_minima/test.csv")
    X_test = torch.Tensor(data_test[["A", "B", "C"]].to_numpy())
    Y_test = torch.Tensor(data_test[["Y"]].to_numpy())
    
    test = []
    
    for i in range(X_test.size()[0]):
        y_correct = Y_test[i, 0]
        
        row = {}
        
        row["A"] = round(X_test[i, 0].item(), 1)
        row["B"] = round(X_test[i, 1].item(), 1)
        row["C"] = round(X_test[i, 2].item(), 1)
        
        row["Y"] = -row["B"]/(2*row["A"])
        
        options = ["a", "b", "c", "d"]
        
        correct_option = np.random.choice(options)
        
        row["correct_answer"] = correct_option
        
        for option in options:
            
            if option == correct_option:
                row["option_%s"%option] = row["Y"]
            else:
                value = y_correct + np.random.normal() * .75

                row["option_%s"%option] = value.item()
            
        test.append(row)
    
    test = pd.DataFrame(test)
    
    test.to_csv("data/polynomial_minima/exam.csv")

def write_model_exam_answers(model):
    data_test = pd.read_csv("data/polynomial_minima/exam.csv")
    X_test = torch.Tensor(data_test[["A", "B", "C"]].to_numpy())
    Y_test = torch.Tensor(data_test[["Y"]].to_numpy())
    option_values = torch.Tensor(data_test[["option_a", "option_b", "option_c", "option_d"]].to_numpy())
    correct_answers = data_test["correct_answer"].to_list()
    
    X_test_seq = repeat_data(X_test, 20)
    
    out, _ = model(X_test_seq)
    
    test = []
    
    for i in range(X_test.size()[0]):
        y_correct = Y_test[i, 0]
        
        row = {}
        
        row["A"] = X_test[i, 0].item()
        row["B"] = X_test[i, 1].item()
        row["C"] = X_test[i, 2].item()
        
        row["Y"] = y_correct.item()
        
        best_distance = float("inf")
        
        best_option = "a"
        
        for option_num, option in enumerate(["a", "b", "c", "d"]):
            
            row["option_%s"%option] = option_values[i, option_num].item()
            
            distance = torch.abs(out[i, 0] - option_values[i, option_num]).item()
            
            if distance < best_distance:
                best_distance = distance
                best_option = option
                
        row["correct_answer"] = correct_answers[i]
                
        row["model_answer"] = best_option
        
        test.append(row)
        
    test = pd.DataFrame(test)
    
    test.to_csv("data/polynomial_minima/exam_answered.csv")
    
    
def get_accuracy():
    exam_answered = pd.read_csv("data/polynomial_minima/exam_answered.csv")
    
    correct_answers = exam_answered["correct_answer"].to_list()
    model_answers = exam_answered["model_answer"].to_list()
    
    accuracy = 0
    
    n = len(correct_answers)
    for i in range(n):
        if correct_answers[i] == model_answers[i]:
            accuracy += 1
            
    accuracy /= n
    
    return accuracy    


def get_reading_comprehension_questions_from_file(path, extra_info=False):
    with open(path, 'r') as file:
        content = file.read()
        content = json.loads(content)
        
        prompt = """Answer the following question with A, B, C, or D about the text below: 
        
        {question}
        
        (A) {option1}
        (B) {option2}
        (C) {option3}
        (D) {option4}
        
        {body}
        """
        
        result = []
        
        for q_num, question in enumerate(content['questions']):
            options = content['options'][q_num]
            body = content['article']
            full_question = prompt.format(question=question, 
                                                option1 = options[0],
                                                option2 = options[1],
                                                option3 = options[2],
                                                option4 = options[3],
                                                body = body)
            
            row = {}
            row["question"] = full_question
            
            if extra_info:
            
                row['correct_answer'] = content["answers"][q_num]
                row['model_answer'] = content["model_answers"][q_num]

                row['question_type'] = "reading_comprehension"
            
            result.append(row)
            
        return result
    
def get_reading_comprehension_questions_from_dir(directory, shuffle=True, n_files=100):
    paths = os.listdir(directory)
    
    if shuffle:
        random.shuffle(paths)
        
    paths = paths[:n_files]
    
    questions = []
    
    for path in paths:
        full_path = "{directory}/{path}".format(directory=directory, path=path)
        questions_for_path = get_reading_comprehension_questions_from_file(full_path)
        
        for q in questions_for_path:
            questions.append(q["question"])
    
    return questions

def get_reading_comprehension_questions(shuffle=True, n_questions=100):
    base_directory = "data/RACE/train"
    
    n = {'high': n_questions // 2, 'middle': n_questions - (n_questions//2)}
    
    result = []
    
    for subdir in ['high', 'middle']:
        directory = "{base}/{subdir}".format(base=base_directory, subdir=subdir)
        
        questions_from_dir = get_reading_comprehension_questions_from_dir(directory, 
                                                                       shuffle=shuffle,
                                                                       n_files=n[subdir])
        random.shuffle(questions_from_dir)
        questions_from_dir = questions_from_dir[:n[subdir]]
        
        result = result + questions_from_dir
    
    return result

def get_reading_comprehension_questions_for_exam():
    filenames = ["middle7023.txt", "middle548.txt", "middle1696.txt", "middle1478.txt"]
    
    result = []
    
    for filename in filenames:
        path = "data/exam/reading_comprehension/%s"%filename
        
        result = result + get_reading_comprehension_questions_from_file(path, extra_info=True)
        
    return result

def format_word_scrambling_question(q):
    index = random.randint(0, 4)
            
    options = []
    for i in range(4):
        if i == index:
            options.append(q['word'])
        else:
            word = q['word']

            while word == q['word']:
                word = list(word)
                random.shuffle(word)
                word = ''.join(word)

            options.append(word)

    q_new = """Which of the following words is the unscrambled version of '{scrambled}'? 

    Please answer with A, B, C, or D.
    (A) {option1}
    (B) {option2}
    (C) {option3}
    (D) {option4}

    """.format(scrambled=q['scrambled'], 
               option1=options[0], 
               option2=options[1],
              option3=options[2],
              option4=options[3])
    
    return q_new

def get_word_scrambling_questions(shuffle=True, n_questions=100):
    path = "data/scrambled_word_dataset.json"
    
    result = []
    
    with open(path) as file:
        questions = json.load(file)
        if shuffle:
            random.shuffle(questions)
        questions = questions[:n_questions]
        
        for q in questions:
            
            q_new = format_word_scrambling_question(q)
            
            result.append(q_new)
    
    return result

def get_word_scrambling_questions_for_exam():
    path = "data/exam/word_scrambling.json"
    
    result = []
        
    with open(path) as file:
        questions = json.load(file)
        
        for q in questions:
            
            row = {}
            
            q_new = format_word_scrambling_question(q)
            
            row['question'] = q_new
            
            row['correct_answer'] = q['correct_answer']
            row['model_answer'] = q['prediction'][1].upper()
            
            row['question_type'] = "word_scrambling"
            
            result.append(row)
    
    return result


def get_polynomial_minima_questions(shuffle=True, n_questions=100):
    result = []
    
    for q_num in range(n_questions):
        
        index = random.randint(0, 4)
        
        a, b, c = np.random.uniform(-5, 5, 3)
        a = abs(a)
        
        a = round(a, 1)
        b = round(b, 1)
        c = round(c, 1)
        
        options = []
        
        for i in range(4):
            if i == index:
                option = -b/(2*a)
            else:
                option = round(np.random.uniform(-5, 5, 1)[0], 3)
                
            options.append(str(round(option, 3)))
                
        question = """Answer the following question with A, B, C, or D. 
        
        What is the minimum of the following polynomial: {A}x^2 + {B}x + {C} ?
        
        (A) {option1}
        (B) {option2}
        (C) {option3}
        (D) {option4}
        """.format(A=a, B=b, C=c, option1=options[0], option2=options[1], option3=options[2], option4=options[3])
        
        result.append(question)
        
    return result


def get_polynomial_minima_questions_for_exam():
    data_test = pd.read_csv("data/polynomial_minima/exam_answered.csv")
    X_test = torch.Tensor(data_test[["A", "B", "C"]].to_numpy())
    Y_test = torch.Tensor(data_test[["Y"]].to_numpy())
    option_values = torch.Tensor(data_test[["option_a", "option_b", "option_c", "option_d"]].to_numpy())
    correct_answers = data_test["correct_answer"].to_list()
    model_answers = data_test["model_answer"].to_list()
    
    result = []
    
    for i in range(X_test.size()[0]):
        
        a, b, c = X_test[i]
        
        a, b, c = round(a.item(), 1), round(b.item(), 1), round(c.item(), 1)
        
        options = list(option_values[i])
        options = [round(option.item(), 2) for option in options]
        
        question = """Answer the following question with A, B, C, or D. 
        
        What is the minimum of the following polynomial: {A}x^2 + {B}x + {C} ?
        
        (A) {option1}
        (B) {option2}
        (C) {option3}
        (D) {option4}
        """.format(A=a, B=b, C=c, option1=options[0], option2=options[1], option3=options[2], option4=options[3])
        
        row = {}
        
        row["question"] = question
        row["correct_answer"] = correct_answers[i].upper()
        row["model_answer"] = model_answers[i].upper()
        
        result.append(row)
        
    return result

def get_exam_questions_for_router():
    n = 1000
    n_scrambling = 334
    n_comprehension = 333
    n_minima = 333
    
    scrambling = get_word_scrambling_questions(n_questions=n_scrambling)
    comprehension = get_reading_comprehension_questions(n_questions=n_comprehension)
    minima = get_polynomial_minima_questions(n_questions=n_minima)
    
    indices = [i for i in range(n)]
    random.shuffle(indices)
    
    questions = scrambling + comprehension + minima
    
    # labels
    # 0 for scrambling
    # 1 for comprehension
    # 2 for minima
    
    labels = [0 for i in range(n_scrambling)] + (
        [1 for i in range(n_comprehension)] + 
        [2 for i in range(n_minima)] )
    
    questions = [questions[index] for index in indices]
    labels = [labels[index] for index in indices]
    
    return questions, labels


def get_exam_questions():
    questions = get_word_scrambling_questions_for_exam() + get_reading_comprehension_questions_for_exam() + (
        get_polynomial_minima_questions_for_exam())
    
    return questions

def get_open_ai_query_for_question(question, secret_key=None):
    formatted_question = question
    formatted_question = formatted_question.replace("\n", " ").replace("  ", "")
    formatted_question = formatted_question.replace("\"", "")
    formatted_question = formatted_question.replace("\'", "")
    query = ""
    query += """curl https://api.openai.com/v1/chat/completions \\"""
    query += "\n"
    query += """-H "Content-Type: application/json" \\"""
    query += "\n"
    query += """-H "Authorization: Bearer {secret_key}" \\""".format(secret_key=secret_key)
    query += "\n"
    query += """-d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": """
    query += """\"{question}\"""".format(question=formatted_question)
    query += """}],
     "temperature": 0.7
   }' > response.txt"""
    return query

def get_open_ai_answer_for_question(question, secret_key=None, num_retries=2):
    
    query = get_open_ai_query_for_question(question, secret_key=secret_key)
    with open("query.sh", 'w') as file:
        file.write(query)
        
    retry_num = 0
    done = False
    
    while not done:
        if retry_num >= num_retries:
            return None
        try:
            subprocess.run(["bash", "query.sh"])
            with open("response.txt", 'r') as file:
                content = json.load(file)
                answer = content['choices'][0]['message']['content']
            done = True
        except KeyError:
            retry_num += 1
            time.sleep(20)
            
    
    return answer

def get_benchmark_answers(questions, secret_key=None):
    result = []
    for question in questions:
        row = question
        row['benchmark_answer'] = get_open_ai_answer_for_question(question['question'], secret_key=secret_key)
        result.append(row)
    return result

def identify_benchmark_answer(response):
    for option in ["A", "B", "C", "D"]:
        if "%s)"%option in response:
            return option
    return None

def get_accuracy_table():
    
    result = []
    
    with open('data/exam/final_exam_result.json', 'r') as file:
        answers = json.load(file)
    
        model_accuracy = {"model": "mixture-of-experts", "word_scrambling": 0, "reading_comprehension": 0, "polynomial_minima": 0, "total": 0}
        benchmark_accuracy = {"model": "gpt-3.5", "word_scrambling": 0, "reading_comprehension": 0, "polynomial_minima": 0, "total": 0}
        
        for answer in answers:
            try:
                answer["question_type"]
            except:
                answer["question_type"] = "polynomial_minima"
                
            if answer["model_answer"] == answer["correct_answer"]:
                model_accuracy["total"] += 1
                model_accuracy[answer["question_type"]] += 1
                
            benchmark_answer = identify_benchmark_answer(answer["benchmark_answer"])
                
            if benchmark_answer == answer["correct_answer"]:
                benchmark_accuracy["total"] += 1
                benchmark_accuracy[answer["question_type"]] += 1
                
        for q_type in ["total", "word_scrambling", "reading_comprehension", "polynomial_minima"]:
            if q_type == "polynomial_minima":
                n_questions = 16
            elif q_type == "total":
                n_questions = 50
            else:
                n_questions = 17
            
            model_accuracy[q_type] = "{acc}%".format(acc=round(100 * model_accuracy[q_type]/n_questions, 2))
            benchmark_accuracy[q_type] = "{acc}%".format(acc=round(100 * benchmark_accuracy[q_type]/n_questions, 2))
        
        result.append(model_accuracy)
        result.append(benchmark_accuracy)

    result = pd.DataFrame(result)
    return result
            
            