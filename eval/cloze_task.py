# Custom query evaluation function with batch inference
def cloze_task_evaluation(model, tokenizer, query_dataset, batch_size=8, temperature=None):
    total = len(query_dataset)
    print('Evaluating query dataset with', total, 'samples')
    correct = 0

    tokenizer.padding_side = "left"

    # Split the queries into batches
    for i in range(0, total, batch_size):
        batch = query_dataset[i:i+batch_size]

        # Extract queries and correct answers for this batch
        queries = [example['query'].strip() for example in batch]
        correct_answers = [example['answers'] for example in batch]

        # Tokenize all queries in the batch
        inputs = tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

        # Generate outputs for all queries in the batch
        if temperature:
            # generate with sampling
            outputs = model.generate(**inputs, max_new_tokens=20, num_return_sequences=1, do_sample=True, temperature=temperature)
        else:
            outputs = model.generate(**inputs, max_new_tokens=20, num_return_sequences=1)

        # Decode the generated sequences
        generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        tails = [pred.strip().split('.')[0].replace(queries[j], '').strip() for j, pred in enumerate(generated_texts)]

        # Check each generated text against the correct answers for the batch
        for tail, correct_ans in zip(tails, correct_answers):
            if tail in correct_ans:
                correct += 1

        print(f"Batch {i} example:")
        print("Query:", queries[0])
        print("Correct answer:", correct_answers[0])
        print("Generated text:", generated_texts[0])
        print("Tail:", tails[0])
        print()

    tokenizer.padding_side = "right"
    accuracy = correct / total
    return {'eval/query_accuracy': accuracy}