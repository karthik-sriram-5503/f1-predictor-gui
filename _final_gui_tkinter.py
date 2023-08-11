#preparing the model
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, BaggingClassifier
import pandas as pd
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
path=r'C:\Users\capvj\Downloads\f1FINAL.csv'
data=pd.read_csv(path)
datax=data.copy()
le_gp=LabelEncoder()
data['GP_name'] = le_gp.fit_transform(data['GP_name'])
le_c=LabelEncoder()

data['constructor'] = le_c.fit_transform(data['constructor'])
le_d=LabelEncoder()
data['driver'] = le_d.fit_transform(data['driver'])
def position_index(x):
    if x<4:
        return 1
    if x>10:
        return 3
    else :
        return 2
X = data.drop(['position'],1)
y = data['position'].apply(lambda x: position_index(x))
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.8,random_state=3)
logistic_regression = LogisticRegression()
random_forest = RandomForestClassifier()
svc = SVC(probability=True)
decision_tree = DecisionTreeClassifier()
knn = KNeighborsClassifier()
naive_bayes = GaussianNB()
logistic_bagging = BaggingClassifier(base_estimator=logistic_regression, n_estimators=10, random_state=42)
random_forest_bagging = BaggingClassifier(base_estimator=random_forest, n_estimators=10, random_state=42)
svc_bagging = BaggingClassifier(base_estimator=svc, n_estimators=10, random_state=42)
decision_tree_bagging = BaggingClassifier(base_estimator=decision_tree, n_estimators=10, random_state=42)
knn_bagging = BaggingClassifier(base_estimator=knn, n_estimators=10, random_state=42)
naive_bayes_bagging = BaggingClassifier(base_estimator=naive_bayes, n_estimators=10, random_state=42)
voting_model = VotingClassifier(estimators=[
    ('logistic', logistic_bagging),
    ('random_forest', random_forest_bagging),
    ('svc', svc_bagging),
    ('decision_tree', decision_tree_bagging),
    ('knn', knn_bagging),
    ('naive_bayes', naive_bayes_bagging)
    ], voting='soft')
voting_model.fit(X_train, y_train)
# creating gui
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter.font import Font

root = tk.Tk()
root.title("Formula One Predictor")
root.geometry("400x500")
root.config(bg='black')

image = Image.open("C:\cool-f1-pictures-60khof1nlpcww408.jpg")
background_image = ImageTk.PhotoImage(image)

container = tk.Frame(root)
container.pack(fill=tk.BOTH, expand=True)

background_label = tk.Label(container, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
my_font = Font(
    family = 'Times',
    size = 30,
    weight = 'bold',
    slant = 'roman',
    underline = 1,
    overstrike = 0
)


heading_label = tk.Label(container, text="FORMULA ONE PREDICTION",bg="red",fg="black",font = my_font)
heading_label.pack(pady=100,anchor=tk.CENTER)
# Sample driver and team names (you can replace these with your own data)
drivers = ['Daniel Ricciardo', 'Kevin Magnussen', 'Carlos Sainz',
                  'Valtteri Bottas', 'Lance Stroll', 'George Russell',
                  'Lando Norris', 'Sebastian Vettel', 'Kimi Räikkönen',
                  'Charles Leclerc', 'Lewis Hamilton', 'Daniil Kvyat',
                  'Max Verstappen', 'Pierre Gasly', 'Alexander Albon',
                  'Sergio Pérez', 'Esteban Ocon', 'Antonio Giovinazzi',
                  'Romain Grosjean','Nicholas Latifi']
teams = ['Renault', 'Williams', 'McLaren', 'Ferrari', 'Mercedes',
                       'AlphaTauri', 'Racing Point', 'Alfa Romeo', 'Red Bull',
                       'Haas F1 Team']
grand_prix =['Albert Park Grand Prix Circuit', 'Sepang International Circuit',
       'Shanghai International Circuit', 'Bahrain International Circuit',
       'Circuit de Barcelona-Catalunya', 'Circuit de Monaco',
       'Istanbul Park', 'Silverstone Circuit', 'Nürburgring',
       'Hungaroring', 'Valencia Street Circuit',
       'Circuit de Spa-Francorchamps', 'Autodromo Nazionale di Monza',
       'Marina Bay Street Circuit', 'Suzuka Circuit',
       'Autódromo José Carlos Pace', 'Yas Marina Circuit',
       'Circuit Gilles Villeneuve', 'Hockenheimring',
       'Korean International Circuit', 'Sochi Autodrom',
       'Baku City Circuit', 'Red Bull Ring', 'Circuit of the Americas',
       'Autódromo Hermanos Rodríguez', 'Circuit Paul Ricard',
       'Buddh International Circuit']
dc={'Daniel Ricciardo':0.942196532,
    'Kevin Magnussen':0.952380952,
    'Carlos Sainz':0.903846154,
    'Valtteri Bottas':0.965034965,
    'Lance Stroll':0.923076923,
    'George Russell':0.958333333,
    'Lando Norris':0.916666667,
    'Sebastian Vettel':0.955,
    'Kimi Räikkönen':0.944099379,
    'Charles Leclerc':0.844444444,
    'Lewis Hamilton':0.945273632,
    'Daniil Kvyat':0.917525773,
    'Max Verstappen':0.914285714,
    'Pierre Gasly':0.93877551,
    'Alexander Albon':0.956521739,
    'Sergio Pérez':0.9333333333,
    'Esteban Ocon':0.923076923,
    'Antonio Giovinazzi':0.88,
    'Romain Grosjean':0.851851852,
    'Nicholas Latifi':0.736547899
    }
cr={'Renault':0.530150754,
    'Williams':0.496259352,
    'McLaren':0.591478697,
    'Ferrari':0.898009950248756,
    'Mercedes':0.877805486,
    'AlphaTauri':0.454773869,
    'Racing Point':0.608478802992518,
    'Alfa Romeo':0.427135678391959,
    'Haas F1 Team':0.343023255813953,
    'Red Bull':0.825
    }
grid=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
driver_var = tk.StringVar()
team_var = tk.StringVar()
gp_var = tk.StringVar()
grid_var = tk.StringVar()
driver_var.set("select your driver")
team_var.set("select the constructor")
gp_var.set("select the grand prix")
grid_var.set("select the starting grid")
driver_menu = tk.OptionMenu(container, driver_var, *drivers)
driver_menu.config(fg='white',bg='red',activeforeground="black")
team_menu = tk.OptionMenu(container, team_var, *teams)
team_menu.config(fg='white',bg='red',activeforeground="black")
gp_menu =  tk.OptionMenu(container , gp_var, *grand_prix)
gp_menu.config(fg='white',bg='red',activeforeground='red')
grid_menu = tk.OptionMenu(container , grid_var, *grid)
grid_menu.config(fg='white',bg='red',activeforeground='red')
driver_menu.pack(pady=20, anchor=tk.CENTER)
team_menu.pack(pady=20, anchor=tk.CENTER)
gp_menu.pack(pady=20, anchor=tk.CENTER)
grid_menu.pack(pady=20, anchor=tk.CENTER)



def on_driver_change(*args):
    global selected_driver
    selected_driver = driver_var.get()
    print("Selected driver:", selected_driver)
    

def on_team_change(*args):
    global selected_team
    selected_team = team_var.get()
    print("Selected team:", selected_team)
    

def on_grandprix_change(*args):
    global selected_grandprix
    selected_grandprix = gp_var.get()
    print("Selected grand prix:", selected_grandprix)
    

def on_grid_change(*args):
    global grid
    grid = grid_var.get()
    print("Seleceted grid :", grid)
    
driver_var.trace("w", on_driver_change)
team_var.trace("w", on_team_change)
gp_var.trace("w", on_grandprix_change)
grid_var.trace("w", on_grid_change)

def predict_output():
    
    driver_confidence = dc[selected_driver]
    constructor_reliablity = cr[selected_team]
    ind={'GP_name':selected_grandprix,'quali_position':grid,'constructor':selected_team,'driver':selected_driver,'driver_confidence':driver_confidence,'constructor_relaiblity':constructor_reliablity}
    Xin=pd.DataFrame([ind])
    Xi = datax.drop('position',1)
    le=LabelEncoder()
    traindata=pd.concat([Xi,Xin],ignore_index=True)    
    traindata['GP_name'] = le.fit_transform(traindata['GP_name'])
    traindata['constructor'] = le.fit_transform(traindata['constructor'])
    traindata['driver'] = le.fit_transform(traindata['driver'])
    Xin=traindata.tail(1)
    result=voting_model.predict(Xin).apply(lambda x: position_index(x))
    print(result)
    if result==[1]:
       finish="Podium Finish"
    elif result ==[2]:
       finish="Points Finish",
    else:
       finish="No Points Finish"
    finissh.set(str(finish))
    label.config(text=f"{finissh.get()}")
      
predict_button = tk.Button(container, text="Predict", command=predict_output,fg='white',bg='red')
predict_button.pack()
finissh=tk.StringVar()

label = tk.Label(container, text=f"{finissh.get()}",fg="white",bg="red")
label.pack(pady=10)
        
def exit_application():
    root.destroy()

exit_button = tk.Button(container, text="Exit", command=exit_application)
exit_button.config(fg='white',bg='red'    )
exit_button.pack(pady=20, anchor=tk.CENTER)



root.mainloop()


