features=['time','outtemp','outhum','outair','innhum','innair']
# print(train[features])
data=train[features]
test_data=test[features]
y_data=train['inntemp']
parameters={'n_estimators':range(10, 300, 10),
                'max_depth':range(2,10,1),
                'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
                'min_child_weight':range(5, 21, 1),
                'subsample':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                'colsample_bytree':[0.5, 0.6, 0.7, 0.8, 0.9, 1],
                'colsample_bylevel':[0.5, 0.6, 0.7, 0.8, 0.9, 1]
                }
    #parameters={'max_depth':range(2,10,1)}
model=xgb.XGBRegressor(seed=25,
                         n_estimators=100,
                         max_depth=3,
                         eval_metric='rmse',
                         learning_rate=0.1,
                         min_child_weight=1,
                         subsample=1,
                         colsample_bytree=1,
                         colsample_bylevel=1,
                         gamma=0)
gs=GridSearchCV(estimator= model,param_grid=parameters,cv=5,refit= True,scoring='neg_mean_squared_error')
gs.fit(data.values,y_data)
print('最优参数: ', gs.best_params_)
y_test= gs.predict(test_data.values)
test['temperature'] = y_test
test[['time','temperature']].to_csv('./baseline_baseline.csv',index=False)