#!/usr/bin/env python
# coding: utf-8

# In[481]:


import pandas as pd
import math
import numpy as np


# In[482]:


data_sa = pd.read_csv('CricketSA.csv')
data_india = pd.read_csv('CricketIndia.csv')
batsman_india_df = pd.read_csv("india_batting_lineup.csv")
batsman_sa_df = pd.read_csv("sa_batting_lineup.csv")


# In[483]:


bowlers_sa = {}
bowlers_india = {}
for i in data_india['Bowler_Name'].unique():
    bowlers_sa[i] = [0,0,0,0,0,0,0]
for i in data_sa['Bowler_Name'].unique():
    bowlers_india[i] = [0,0,0,0,0,0,0]
    
batsman_india = {}
batsman_sa = {}
for i in batsman_india_df['India Batting Lineup (Batting First)']:
    batsman_india[i] = [0,0,0,0,0]
for i in batsman_sa_df['SA Batting Lineup (Batting Second)']:
    batsman_sa[i] = [0,0,0,0,0]


# In[484]:


batting_order_sa = batsman_sa_df['SA Batting Lineup (Batting Second)'].values.tolist()
batting_order_india = batsman_india_df['India Batting Lineup (Batting First)'].values.tolist()


# In[485]:


#INDIA BATTING
maiden_flag = 0
over_count = -1

on_strike = batsman_india[batting_order_india[0]]
off_strike = batsman_india[batting_order_india[1]]

extras = 0

playing_batsman = [0,1]

for x,y in data_india.iterrows():
        
    current_ball_runs = y[2]
    
    print(on_strike)
    
    #EXTRA RUNS
    if(y[6])!=0:
        current_ball_runs += y[6]
    
    #check over
    if(over_count)==-1:
        bowlers_sa[y[0]][0] += 1
        over_count += 1
        
    if(over_count)!=math.floor(y[1]):
        over_count = math.floor(y[1])
        
        on_strike,off_strike = off_strike, on_strike
        
        bowlers_sa[y[0]][0] += 1

        #maiden 
        if(maiden_flag==1):
              bowlers_sa[y[0]][1] += 1
        
        
    #increase ball count
    on_strike[1] += 1
    
    
    #wide ball
    if(y[5] == 1):
        on_strike[1] -= 1
        bowlers_sa[y[0]][5] += 1
        extras += 1
        maiden_flag = 0
        
    #no ball
    if(y[4] == 1):
        bowlers_sa[y[0]][4] += 1
        extras += 1
        maiden_flag = 0
        
    #check wicket
    if(y[3])==1:
        
        on_strike[4] = (on_strike[0]/on_strike[1])*100
        
        on_strike = batsman_india[batting_order_india[max(playing_batsman)+1]]
        
        playing_batsman.append(max(playing_batsman)+1)
        
        wordlist = ['run out','runout','Runout',"Run Out","run-out","RUN OUT", "Run out"]
        if(any(substring not in y[7] for substring in wordlist)):
            bowlers_sa[y[0]][3] += 1
        print("\n\nWICKET\n\n")
        
        
    if(current_ball_runs%2==0):
        on_strike[0] += current_ball_runs
        bowlers_sa[y[0]][2] += current_ball_runs
        if(current_ball_runs>0):
            maiden_flag = 0
        if(current_ball_runs==4):
            on_strike[2] += 1
        if(current_ball_runs==6):
            on_strike[3] += 1
            
    else:
        on_strike[0] += current_ball_runs
        bowlers_sa[y[0]][2] += current_ball_runs
        
        maiden_flag = 0
        
        on_strike,off_strike = off_strike, on_strike
        
    print("Ball ",y[1], "Runs",current_ball_runs)    
    print("Onstrike",on_strike)
    print("Offstrike",off_strike)    
     
on_strike[4] = (on_strike[0]/on_strike[1])*100        

off_strike[4] = (off_strike[0]/off_strike[1])*100     
        
for bowler in bowlers_sa:
    bowlers_sa[bowler][6] = bowlers_sa[bowler][2]/bowlers_sa[bowler][0]
    print(bowler)
    
score_india = 0
score_india += extras

for i in batsman_india:
    score_india += (batsman_india[i][0])

print("FINAL SCORE ", score_india)    
        
    


# In[486]:


print(bowlers_sa)


# In[487]:


print(batsman_india)


# In[488]:


#SA BATTING
maiden_flag = 0
over_count = -1

on_strike = batsman_sa[batting_order_sa[0]]
off_strike = batsman_sa[batting_order_sa[1]]

extras = 0

playing_batsman = [0,1]

for x,y in data_sa.iterrows():
        
    current_ball_runs = y[2]

    #EXTRA RUNS
    if(y[6])!=0:
        current_ball_runs += y[6]
    
    #check over
    if(over_count)==-1:
        bowlers_india[y[0]][0] += 1
        over_count += 1
        
    if(over_count)!=math.floor(y[1]):
        over_count = math.floor(y[1])
        
        on_strike,off_strike = off_strike, on_strike
        
        bowlers_india[y[0]][0] += 1

        #maiden 
        if(maiden_flag==1):
              bowlers_india[y[0]][1] += 1
        
        
    #increase ball count
    on_strike[1] += 1
    
    
    #wide ball
    if(y[5] == 1):
        on_strike[1] -= 1
        bowlers_india[y[0]][5] += 1
        extras += 1
        maiden_flag = 0
        
    #no ball
    if(y[4] == 1):
        bowlers_india[y[0]][4] += 1
        extras += 1
        maiden_flag = 0
        
    #check wicket
    if(y[3])==1:
        
        on_strike[4] = (on_strike[0]/on_strike[1])*100
        
        on_strike = batsman_sa[batting_order_sa[max(playing_batsman)+1]]
        
        playing_batsman.append(max(playing_batsman)+1)
        
        wordlist = ['run out','runout','Runout',"Run Out","run-out","RUN OUT", "Run out"]
        if(any(substring not in y[7] for substring in wordlist)):
            bowlers_india[y[0]][3] += 1
        print("\n\nWICKET\n\n")
        
        
    if(current_ball_runs%2==0):
        on_strike[0] += current_ball_runs
        bowlers_india[y[0]][2] += current_ball_runs
        if(current_ball_runs>0):
            maiden_flag = 0
        if(current_ball_runs==4):
            on_strike[2] += 1
        if(current_ball_runs==6):
            on_strike[3] += 1
            
    else:
        on_strike[0] += current_ball_runs
        bowlers_india[y[0]][2] += current_ball_runs
        
        maiden_flag = 0
        
        on_strike,off_strike = off_strike, on_strike
        
    print("Ball ",y[1], "Runs",current_ball_runs)    
    print("Onstrike",on_strike)
    print("Offstrike",off_strike)    
     
on_strike[4] = (on_strike[0]/on_strike[1])*100        

off_strike[4] = (off_strike[0]/off_strike[1])*100     
        
for bowler in bowlers_india:
    bowlers_india[bowler][6] = bowlers_india[bowler][2]/bowlers_india[bowler][0]
    print(bowler)
    
score_sa = 0
score_sa += extras

for i in batsman_sa:
    score_sa += (batsman_sa[i][0])

print("FINAL SCORE ", score_sa)    
        
    


# In[489]:


bowlers_india


# In[490]:


batsman_sa


# In[491]:


batsman = {}
for player in batsman_india.keys():
    batsman[player] = 0
for player in batsman_sa.keys():
    batsman[player] = 0


# In[492]:


for i in batsman_india:
    batsman[i] += batsman_india[i][0]
    batsman[i] += batsman_india[i][2]*2
    batsman[i] += batsman_india[i][3]*6
    if(batsman_india[i][4])<100:
        batsman[i] -= 5 
    elif (batsman_india[i][4])==100:
        batsman[i] += 0
    elif (batsman_india[i][4]>100 and batsman_india[i][4]<=200):
        batsman[i] += 5 
    elif (batsman_india[i][4]>200 and batsman_india[i][4]<=300):
        batsman[i] += 10 
    elif (batsman_india[i][4]>300 and batsman_india[i][4]<=400):
        batsman[i] += 15
    else:
        batsman[i] += 15
    
for i in batsman_sa:
    batsman[i] += batsman_sa[i][0]
    batsman[i] += batsman_sa[i][2]*2
    batsman[i] += batsman_sa[i][3]*6
    if(batsman_sa[i][4])<100:
        batsman[i] -= 5 
    elif (batsman_sa[i][4])==100:
        batsman[i] += 0
    elif (batsman_sa[i][4]>100 and batsman_sa[i][4]<=200):
        batsman[i] += 5 
    elif (batsman_sa[i][4]>200 and batsman_sa[i][4]<=300):
        batsman[i] += 10 
    elif (batsman_sa[i][4]>300 and batsman_sa[i][4]<=400):
        batsman[i] += 15
    else:
        batsman[i] += 15
    


# In[493]:


best_batsman = max(batsman, key=batsman.get)


# In[494]:


bowler = {}
for player in bowlers_india.keys():
    bowler[player] = 0
for player in bowlers_sa.keys():
    bowler[player] = 0


# In[495]:


for i in bowlers_india:
    bowler[i] += bowlers_india[i][3]*10
    if(bowlers_india[i][6])<6:
        bowler[i] += 20 
    elif (bowlers_india[i][6])==6:
        bowler[i] += 15
    elif (bowlers_india[i][6]>6 and bowlers_india[i][6]<=7):
        bowler[i] += 10
    elif (bowlers_india[i][6]>7 and bowlers_india[i][6]<=8):
        bowler[i] += 5 
    elif (bowlers_india[i][6]>8 and bowlers_india[i][6]<=9):
        bowler[i] += 0
    elif (bowlers_india[i][6]>9 and bowlers_india[i][6]<=10):
        bowler[i] -= 5
    else:
        bowler[i] -= 10
        
        
for i in bowlers_sa:
    bowler[i] += bowlers_sa[i][3]*10
    if(bowlers_sa[i][6])<6:
        bowler[i] += 20 
    elif (bowlers_sa[i][6])==6:
        bowler[i] += 15
    elif (bowlers_sa[i][6]>6 and bowlers_sa[i][6]<=7):
        bowler[i] += 10
    elif (bowlers_sa[i][6]>7 and bowlers_sa[i][6]<=8):
        bowler[i] += 5 
    elif (bowlers_sa[i][6]>8 and bowlers_sa[i][6]<=9):
        bowler[i] += 0
    elif (bowlers_sa[i][6]>9 and bowlers_sa[i][6]<=10):
        bowler[i] -= 5
    else:
        bowler[i] -= 10
        
    


# In[496]:


best_bowler = max(bowler, key=bowler.get)


# In[497]:


bowler


# In[498]:


if (bowler[best_bowler]>batsman[best_batsman]):
    man_of_the_match = best_bowler
else:
    man_of_the_match = best_batsman


# In[499]:


final = {}
if(score_india>score_sa):
    final["Result"] = "India won the match by "+str(score_india - score_sa)+" runs"
else:
    final["Result"] = "India won the match by "+str(score_sa - score_india)+" runs"
    
    
final["Best Batsman"] = best_batsman
final["Best Bowler"] = best_bowler
final["Man of the Match"] = man_of_the_match
print(final)


# In[500]:



CSV = ",Runs Scored,Ball Faced,4s,6s,Strike Rate\n"
for k,v in batsman_india.items():
    line = "{},{}\n".format(str(k), "".join(','.join(map(str, v))))
    CSV+=line
#print (CSV) 


CSV += "\n,Over,Maiden,Run,Wickets,NB,WD,Economy rate\n"
for k,v in bowlers_sa.items():
    line = "{},{}\n".format(str(k), "".join(','.join(map(str, v))))
    CSV+=line
#print (CSV) 

CSV += "\n,Runs Scored,Ball Faced,4s,6s,Strike Rate\n"
for k,v in batsman_sa.items():
    line = "{},{}\n".format(str(k), "".join(','.join(map(str, v))))
    CSV+=line
#print (CSV) 


CSV += "\n,Over,Maiden,Run,Wickets,NB,WD,Economy rate\n"
for k,v in bowlers_india.items():
    line = "{},{}\n".format(str(k), "".join(','.join(map(str, v))))
    CSV+=line
#print (CSV) 

CSV += "\n"
for k,v in final.items():
    line = "{},{}\n".format(k, "".join(v))
    CSV+=line
print (CSV)

with open("Output.csv", "w") as text_file:
    text_file.write(CSV)


# In[501]:


pd_india_batsman = pd.DataFrame.from_dict(batsman_india).transpose()
pd_sa_batsman = pd.DataFrame.from_dict(batsman_sa).transpose()
pd_india_bowlers = pd.DataFrame.from_dict(bowlers_india).transpose()
pd_sa_bowlers = pd.DataFrame.from_dict(bowlers_sa).transpose()


# In[502]:


pd_india_batsman.columns = ["Runs Scored","Ball Faced","4s","6s","Strike Rate"]
pd_sa_batsman.columns = ["Runs Scored","Ball Faced","4s","6s","Strike Rate"]
pd_india_bowlers.columns = ["Over","Maiden","Run","Wickets","NB","WD","Economy rate"]
pd_sa_bowlers.columns = ["Over","Maiden","Run","Wickets","NB","WD","Economy rate"]


# In[503]:


pd_write = pd.concat([pd.concat([pd_india_batsman, pd_sa_bowlers,pd_sa_batsman,pd_sa_bowlers], axis=0,sort = False)]).to_csv('combined.csv')


# In[504]:


pd_india_batsman.to_csv("1. Innings 1 - batting")
pd_sa_bowlers.to_csv("2. Innings 1 - balling")
pd_sa_batsman.to_csv("3. Innings 2 - batting")
pd_india_batsman.to_csv("4. Innings 2 - balling")


# In[ ]:




