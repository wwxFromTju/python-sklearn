#!/usr/bin/env python
# encoding=utf-8

import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

results = pd.read_csv('data/leagues_NBA_2014_games_games.csv', parse_dates=True, skiprows=[0,])
results.columns = ["Date", 'Score Type', 'Visitor Team', 'VisitorPts', 'Home Team', 'HomePts', 'OT?', 'Notes']

results["HomeWin"] = results['VisitorPts'] < results['HomePts']
y_true = results['HomeWin'].values

print('Home Win percentage: {0:.1f}%'.format(100 * results['HomeWin'].sum() / results['HomeWin'].count()))

results['HomeLastWin'] = False
results['VisitorLastWin'] = False

won_last = defaultdict(int)

for index, row in results.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    row['HomeLastWin'] = won_last[home_team]
    row['VisitorLastWin'] = won_last[visitor_team]
    results.ix[index] = row

    won_last[home_team] = row['HomeWin']
    won_last[visitor_team] = not row['HomeWin']

clf = DecisionTreeClassifier(random_state=14)

X_previouswins = results[['HomeLastWin', 'VisitorLastWin']].values
scores = cross_val_score(clf, X_previouswins, y_true, scoring='accuracy')
print('\nUsing just the last result from the home and visitor teams')
print('Accuracy: {0:.1f}%'.format(np.mean(scores) * 100))

results["HomeWinStreak"] = 0
results["VisitorWinStreak"] = 0
win_streak = defaultdict(int)

for index, row in results.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    row["HomeWinStreak"] = win_streak[home_team]
    row["VisitorWinStreak"] = win_streak[visitor_team]
    results.ix[index] = row
    if row["HomeWin"]:
        win_streak[home_team] += 1
        win_streak[visitor_team] = 0
    else:
        win_streak[home_team] = 0
        win_streak[visitor_team] += 1

X_winstreak =  results[["HomeLastWin", "VisitorLastWin", "HomeWinStreak", "VisitorWinStreak"]].values
scores = cross_val_score(clf, X_winstreak, y_true, scoring='accuracy')
print("\nUsing whether the home team is ranked higher")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

ladder = pd.read_csv("data/leagues_NBA_2013_standings_expanded-standings.csv")
results["HomeTeamRanksHigher"] = 0
for index, row in results.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    home_rank = ladder[ladder["Team"] == home_team]["Rk"]
    visitor_rank = ladder[ladder["Team"] == visitor_team]["Rk"]
    row["HomeTeamRanksHigher"] = int(home_rank.values[0] > visitor_rank.values[0])
    results.ix[index] = row

X_homehigher =  results[["HomeLastWin", "VisitorLastWin", "HomeTeamRanksHigher"]].values
scores = cross_val_score(clf, X_homehigher, y_true, scoring='accuracy')
print("\nUsing whether the home team is ranked higher")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

parameter_space = {
                   "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                   }
grid = GridSearchCV(clf, parameter_space)
grid.fit(X_homehigher, y_true)
print("\nAccuracy: {0:.1f}%".format(grid.best_score_ * 100))


last_match_winner = defaultdict(int)
results["HomeTeamWonLast"] = 0

for index, row in results.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    teams = tuple(sorted([home_team, visitor_team]))  # Sort for a consistent ordering
    # Set in the row, who won the last encounter
    row["HomeTeamWonLast"] = 1 if last_match_winner[teams] == row["Home Team"] else 0
    results.ix[index] = row
    # Who won this one?
    winner = row["Home Team"] if row["HomeWin"] else row["Visitor Team"]
    last_match_winner[teams] = winner

X_home_higher =  results[["HomeTeamRanksHigher", "HomeTeamWonLast"]].values
scores = cross_val_score(clf, X_home_higher, y_true, scoring='accuracy')
print("\nUsing whether the home team is ranked higher")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

encoding = LabelEncoder()
encoding.fit(results["Home Team"].values)
home_teams = encoding.transform(results["Home Team"].values)
visitor_teams = encoding.transform(results["Visitor Team"].values)
X_teams = np.vstack([home_teams, visitor_teams]).T

onehot = OneHotEncoder()
X_teams = onehot.fit_transform(X_teams).todense()

scores = cross_val_score(clf, X_teams, y_true, scoring='accuracy')
print("\nAccuracy: {0:.1f}%".format(np.mean(scores) * 100))

clf = RandomForestClassifier(random_state=14)
scores = cross_val_score(clf, X_teams, y_true, scoring='accuracy')
print("\nUsing full team labels is ranked higher")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

X_all = np.hstack([X_home_higher, X_teams])
scores = cross_val_score(clf, X_all, y_true, scoring='accuracy')
print("\nUsing whether the home team is ranked higher")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

parameter_space = {
                   "max_features": [2, 10, 'auto'],
                   "n_estimators": [100,],
                   "criterion": ["gini", "entropy"],
                   "min_samples_leaf": [2, 4, 6],
                   }
clf = RandomForestClassifier(random_state=14)
grid = GridSearchCV(clf, parameter_space)
grid.fit(X_all, y_true)
print("\nAccuracy: {0:.1f}%".format(grid.best_score_ * 100))
print(grid.best_estimator_)
