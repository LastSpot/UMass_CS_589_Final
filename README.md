# UMass_CS_589_Final

Running the code:
In main.py, several lines that are commented out suggest the way of calling a specific algorithm on a specific dataset, uncomment lines and run main.py to see results.

For digits: line 17-19
For titanic: line 33-35
For loan: line 49-51
For parkinson: line 63-65

Notes:
ALL algorithms used belongs to Minh Le, this is due to the fact that Minh's implementation mostly used classes, and is a lot easier to generalize on new data. This was discussed and agreed on for both partners

During the implementation, serveral attributes were excluded, loan_id was excluded from the loan data set as an identifier should play no role in whether someone receives a loan or not. Similarly, name was excluded from the titanic dataset as there is no reason a name should play a role in whether someone survived or not. Excluding these attributes benefits the machine leaning algorithmns as they prevent overfitting. 