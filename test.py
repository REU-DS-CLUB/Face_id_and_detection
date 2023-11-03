def prob_game_over(sp, sv, b, memo):
    # Если игра уже должна была завершиться
    if abs(sp - sv) >= b:
        return 0
    
    # Если текущие суммы уже были рассмотрены ранее
    if (sp, sv) in memo:
        return memo[(sp, sv)]

    probability = 0
    for next_num_p in range(1, 11):
        for next_num_v in range(1, 11):
            new_sp = sp + next_num_p
            new_sv = sv + next_num_v
            
            # Если игра закончится на этом ходу
            if abs(new_sp - new_sv) >= b:
                probability += 0.01  # 0.1 * 0.1 = 0.01
            else:
                probability += 0.01 * prob_game_over(new_sp, new_sv, b, memo)

    memo[(sp, sv)] = probability
    return probability

def expected_rounds(b):
    memo = {}
    expectation = 0
    round_num = 1

    while True:
        # Вероятность завершения игры на этом раунде
        game_over_probability = prob_game_over(0, 0, b, memo)
        expectation += round_num * game_over_probability

        if game_over_probability < 1e-9:  # Приближение к 0 для окончания цикла
            break
        round_num += 1

    return expectation

b = 1
print(f"Математическое ожидание количества раундов: {expected_rounds(b)}")
