1 bg 250:fg 0:cls
5 b = 0:cp = 0
9 for j = 1 to 20
10 for i = 0 to 95
11 if b = 1 then bg 255 else bg 250
12 b = 1 - b:cp = cp + 1
13 if cp = 60 then cp = 0:b = 1 - b
20 print chr$(i + 32);
30 next i:next j
