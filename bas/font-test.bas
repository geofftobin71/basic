1 bg 12:fg 0:cls
5 b = 0
10 for i = 32 to 127
11 if b = 1 then bg 12 else bg 15
12 b = 1 - b
13 if i = 40 or i = 80 then b = 1 - b
20 print chr$(i);
30 next i
