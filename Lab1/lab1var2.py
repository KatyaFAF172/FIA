import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

print('\nWelcome to Luna City! Please answer the following questions so the our system can identify you.\n')
df = pd.read_csv('touristData.csv')

print (df.columns)
	
questions = ['1. What calendar are you using?\n\n', '2.What is your mother tongue?\n\n', '3.What is precious to you?\n', '4.How do you define your gait?\n', '5.Your clothes?\n', '6.Your height?\n']
def check(n, answ, df):
    n_col = len(df.columns)
    z = 1
    looni = 0
    while z < n_col:
        if n+1 == z:
            newdf = df.loc[df[df.columns[z]] == answ]
            print (newdf) #to see if input value that is in DB refers to tourist or looni
            if newdf['Tourist'].values == 0:
                looni += 0
                print('looni--', looni)
            else:
                looni += 16
                print('tourist--', looni)
        z += 1
    return looni

list_all_answers=[]
for q in range(len(questions)):
    answ = input(questions[q])
    list_all_answers.append(check(q, answ, df))
	#check(q,answ, df)
    
print('\nYou are ', sum(list_all_answers),'% tourist. Welcome to the Luna City!')