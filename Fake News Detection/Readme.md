The main objective is to detect the fake news, which is a classic text classification problem with a straight forward proposition. 
Using sklearn, a TfidfVectorizer was build and then, initializes PassiveAggressive Classifier and fit the model.

OBJECTIVE

The main objective is to detect the fake news, which is a classic text classification problem with a straight forward proposition. 
It is needed to build a model that can differentiate between “Real” news and “Fake” news. Using sklearn, we build a TfidfVectorizer on our dataset. 
Then, we initialize a PassiveAggressive Classifier and fit the model. In the end, the accuracy score and the confusion matrix tell us how well our model fares.

PROJECT DESCRIPTION

To get the statistics about the news, we need to count the appearance of the word in the document. But one issue with word counting is that words 
like ‘the’ appears many times in the document but its count is not meaningful in encoded vector.
One solution for this is to count the word frequency. The method used for this is TF-IDF which stands for “Term Frequency – Inverse Document Frequency “.

•	Term Frequency: It indicates how many times the word appears in the document. A higher value means the word appears more times and so on.
•	Inverse Document Frequency: IDF measures how significant the term is in other articles of the same writer. Words that occur many times in a 
  document may occur many times in others also.
•	Passive Aggressive algorithms: They are online learning algorithms. Such an algorithm remains passive for a correct classification outcome, and turns 
  aggressive in the event of a miscalculation, updating and adjusting.

In short, TF-IDF is a word frequency counter that tries to highlight the interesting words. TF-IDF tokenize the document and encode the new document. 
TF-IDF Vectorizer converts the raw data in the document into TF-IDF matrix.

THE DATASET

The dataset we used for this python project is news.csv. This dataset has a shape of 6335×4. The first column identifies the news, the second and third are 
the title and text, and the fourth column has labels denoting whether the news is REAL or FAKE.
