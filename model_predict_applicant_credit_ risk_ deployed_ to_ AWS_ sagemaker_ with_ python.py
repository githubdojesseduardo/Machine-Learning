import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import sagemaker
from sagemaker.xgboost import XGBoost
from sagemaker.model_monitor import DefaultModelMonitor


data = pd.read_csv('credit_data.csv')


data.fillna(method='ffill', inplace=True)
data = pd.get_dummies(data)



X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
print(f'Accuracy: {accuracy}, AUC: {auc}')


sagemaker_session = sagemaker.Session()
role = 'your-aws-role'

xgb = XGBoost(entry_point='train.py', role=role, instance_count=1, instance_type='ml.m4.xlarge', framework_version='1.2-1')
xgb.fit({'train': 's3://your-bucket/train_data.csv'})
predictor = xgb.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')


monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    volume_size_in_gb=100,
    max_runtime_in_seconds=3600,
)
monitor.create_monitoring_schedule(
    endpoint_input=predictor.endpoint,
    output_s3_uri='s3://your-bucket/monitoring',
    schedule_cron_expression='cron(0 * ? * * *)'
)
