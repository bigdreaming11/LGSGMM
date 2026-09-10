def single_scores(boxes0 ,boxes1,matches ):
    match_t=matches#tensor
    GA=0#真值匹配数
    for box0 in boxes0:
        for box1 in boxes1:
            if box0[0]==box1[0] and box0[1]==box1[1]:
                GA+=1
    TA=0#正确匹配
    FA=0#错误匹配
    for match in match_t:
        id0=match[0]
        id1=match[1]
        if boxes0[id0][0]==boxes1[id1][0] and boxes0[id0][1]==boxes1[id1][1] :
        # if  boxes0[id0][1]==boxes1[id1][1] :
            TA+=1
        else:
            FA+=1
    #统计漏匹配MA
    MA=GA-TA
    # MA=GA-TA
    epsilon = 1e-8
    scores = TA / (GA + FA + MA+ epsilon )
    precision = TA / (FA + TA + epsilon)
    recall=TA/(GA+epsilon)
    # precision=0
    # recall=0
    return scores,precision,recall