# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 11:52:26 2019

@author: mingjay
"""

作業
閱讀以下兩篇文獻，了解隨機森林原理，並試著回答後續的思考問題

隨機森林 (random forest) - 中文
how random forest works - 英文
1. 隨機森林中的每一棵樹，是希望能夠

Ans: 不要過度生長，避免 Overfitting

2. 假設總共有 N 筆資料，每棵樹用取後放回的方式抽了總共 N 筆資料生成，請問這棵樹大約使用了多少 % 不重複的原資料生成? 

Ans: 
考慮N很大的情形:    
N筆資料中, 特定一筆資料A單次被抽中的機率是(小寫p)pA=1/N. 也就是A單次不被抽到的機率是p = (1-pA).
經過N次抽取, 特定資料A都不被抽到的機率是(大寫P)P = (1-pA)^N

一些數學: 
P = (1-1/N)^N =  [(N-1)/N]^N
變數變換 令 N-1 = X
--> P = [X/(X+1)]^(X+1) = [X/(X+1)]^X * X/(X+1)
已知 e^1 = (1+1/X)^X when x-->infinite
-->P = (1/e) * 1 when x-->infinite p.s. X/(X+1)-->1 when X-->infinite
-->P ~ 0.368 

考慮題目: N次抽取得到N筆特定資料的機率=N筆特定資料都被抽到的機率, 那就是(1-P) = 0.632 定義為PA, 也就是63.2%
   

