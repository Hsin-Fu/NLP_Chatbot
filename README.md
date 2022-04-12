# Practice
NLP practice
By text, decide do people need help or not

# Loading Data

```py
chatbot = pd.read_csv('Sheet_1.csv',
                      usecols=['response_id','class','response_text'],
                      encoding='latin-1')
```

not_flagged => 0 flagged => 1

```py
chatbot["class"] = [1 
                 if each == "flagged" 
                 else 0 
                 for each in chatbot["class"]]
```
# Model
```py
chatbot['Label'] = chatbot['class']

x = chatbot.response_text
y = chatbot.Label

#split train set and test set
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1)
vect = CountVectorizer()

x_train_dtm = vect.fit_transform(x_train)
x_test_dtm = vect.transform(x_test)

from sklearn.ensemble import RandomForestClassifier

#Random Forest
rf = RandomForestClassifier(max_depth=10,
                            max_features=10)

rf.fit(x_train_dtm,y_train)

rf_predict = rf.predict(x_test_dtm)

```
# Example
```py
input_text = 'Had a friend open up to me about his mental addiction to weed and how it was taking over his life and making him depressed'

predict = 'need help'
```

```py
input_text = 'Hello world!'

predict = 'do not need help'
```
