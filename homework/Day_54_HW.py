# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 21:07:00 2019

@author: mingjay
"""

作業
試著想想看, 非監督學習是否有可能使用評價函數 (Metric) 來鑑別好壞呢?
(Hint : 可以分為 "有目標值" 與 "無目標值" 兩個方向思考)

有⽬目標值的分群•如果資料有⽬目標值，只是先忽略略⽬目標值做非監督學習，則只要微調後，就可以使⽤用原本監督的測量量函數評估準確性•
無⽬目標值的分群•但通常沒有⽬目標值/⽬目標值非常少才會⽤用非監督模型，這種情況下，只能使⽤用資料本⾝身的分布資訊，來來做模型的評估
