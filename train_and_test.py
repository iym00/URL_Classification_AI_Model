import pandas as pds
import numpy as np
from KNN import KNN
import re
import urllib.parse
import ipaddress
from sklearn.model_selection import train_test_split

# read in dataset to train ai model

dataset = pds.read_csv('Webpages_Classification_test_data_copy.csv')

# The dataset has the label values as "good" (benign page) or "bad" (malicious page). KNN only works with numerical datasets so conversion is done.

dataset['label'] = dataset['label'].map({'good': 1, 'bad': 0})

# The dataset has the https column values as "yes" (uses https) or "no" (doesn't use https). KNN only works with numerical datasets so conversion is done.

dataset['https'] = dataset['https'].map({'yes': 1, 'no': 0})

# The dataset has the who_is column values as "complete" (if WHO IS domain info is complete) or "incomplete" (if the WHO IS domain info is not complete). KNN only works with numerical datasets so conversion is done.

dataset['who_is'] = dataset['who_is'].map({'complete': 1, 'incomplete': 0})

# This dataset includes a url feature column. This function converts it to a set of numerical feature columns:

def convert_url_to_numericalVALS(url):

	parsed = urllib.parse.urlparse(str(url))

	domain = parsed.netloc

	path = parsed.path

	query = parsed.query

	new_features = {}

	new_features['number_of_digits_in_url'] = sum(character.isdigit() for character in str(url))

	new_features['number_of_special_chars_in_url'] = len(re.findall(r'[^\w]', str(url)))

	new_features['num_subdomains_in_url'] = domain.count('.') - 1

	new_features['count_https'] = url.lower().count('https')

	new_features['path_length'] = len(path)

	new_features['query_length'] = len(query)

	s_k = ['login', 'secure', 'update', 'admin', 'verify', 'account']

	new_features['number_of_sus_words_in_url'] = int(any(word in url.lower() for word in s_k))

	return new_features

	 

# Make numerical columns from the new numerical features created from the url text column to replace the url text column later.

numerical_features_of_url = dataset['url'].apply(convert_url_to_numericalVALS).apply(pds.Series)

dataset = pds.concat([dataset, numerical_features_of_url], axis=1)

# This dataset includes a content feature column. This function converts it to a set of numerical feature columns:

def convert_content_to_numericalVALS(content):

	str_version_of_page_content = str(content)

	new_features = {}

	new_features['page_content_length'] = len(str_version_of_page_content)

	new_features['number_of_words_in_page'] = len(str_version_of_page_content.split())

	new_features['number_of_digits_in_page'] = sum(character.isdigit() for character in str_version_of_page_content)

	new_features['number_of_special_characters_in_page'] = len(re.findall(r'[^\w\s]', str_version_of_page_content))

	s_w = ['login', 'logon', 'signin', 'signon', 'verify', 'auth', 'authenticate', 'reauthenticate', 'account', 'userid', 'username', 'password', 'credentials', 'identity', 'profile', 'urgent', 'immediately', 'asap', 'warning', 'alert', 'risk', 'suspended', 'locked', 'unauthorized', 'limit', 'failure', 'billing', 'invoice', 'payment', 'creditcard', 'cardnumber', 'bank', 'deposit', 'withdraw', 'paypal', 'update', 'confirm', 'reset', 'upgrade', 'change', 'reactivate', 'renew', 'amazon', 'paypal', 'apple', 'microsoft', 'google', 'bankofamerica', 'chase', 'netflix', 'irs', 'gov', 'secure', 'support']

	new_features['number_of_sus_words_in_page'] = int(any(word in str_version_of_page_content.lower() for word in s_w))

	new_features['num_of_script_tags_in_page'] = int('<script' in str_version_of_page_content.lower())

	new_features['num_of_eval_calls_in_page'] = int('eval(' in str_version_of_page_content.lower())

	new_features['num_of_iframes_in_page'] = int('<iframe' in str_version_of_page_content.lower())

	new_features['num_of_onclicks_in_page'] = int('onclick=' in str_version_of_page_content.lower())

	new_features['number_of_sus_anchor_tag_usages_in_page'] = int(bool(re.search(r'<a[^>]*href=[\'"]?javascript:', str_version_of_page_content.lower())))

	return new_features

# Create numerical features columns from text feature columns using the function and add them to the dataset.

numerical_features_of_page_content = dataset['content'].apply(convert_content_to_numericalVALS).apply(pds.Series)

dataset = pds.concat([dataset, numerical_features_of_page_content], axis=1)

# This dataset includes a ip_address feature column. This function converts it to a numerical feature column:

def ip_to_32bitINT(ip):
	return int(ipaddress.IPv4Address(ip))

# Add a new feature column that represents the integer representation of the ip addresses (ip address feature columns dropped later on along with a few others).

dataset['ip_add_as_int'] = dataset['ip_add'].apply(ip_to_32bitINT)

# Convert tlds into numeric values and add a new feature column representing the numeric values of the tld to the dataset.

top_tlds = dataset['tld'].value_counts().nlargest(10).index

dataset['tld_limited'] = dataset['tld'].where(dataset['tld'].isin(top_tlds), other='other')

tld_dummies = pds.get_dummies(dataset['tld_limited'], prefix='tld').astype(int)

dataset = pds.concat([dataset, tld_dummies], axis=1)

# Convert geo_loc to numeric value and add that numeric feature column representing the geo_loc of the url to the dataset.

top_countries = dataset['geo_loc'].value_counts().nlargest(10).index

dataset['geo_loc_limited'] = dataset['geo_loc'].where(dataset['geo_loc'].isin(top_countries), other='other')

country_dummies = pds.get_dummies(dataset['geo_loc_limited'], prefix='country').astype(int)

dataset = pds.concat([dataset, country_dummies], axis=1)

# Drop all unnecessary columns.

dataset.drop(columns=['ip_add'], inplace=True)

dataset.drop(columns=['content'], inplace=True)

dataset.drop(columns=['url'], inplace=True)

dataset.drop(columns=['tld'], inplace=True)

dataset.drop(columns=['geo_loc'], inplace=True)

dataset.drop(columns=['tld_limited'], inplace=True)

dataset.drop(columns=['geo_loc_limited'], inplace=True)

# Separate dataset into input and output columns (feature and label columns) and convert dataframes into numpy arrays. 

X = dataset.drop(columns=['label']).to_numpy()

y = dataset['label'].to_numpy()

# Train the model with the prepared dataset.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # we are using 20% of the dataset to test the model and 80% to train it.
)

KNN_Object = KNN(k=5)

print("Training dataset features columns:")

print("\n")

print(X_train)

print("\n")

print("Training dataset label column: ")

print("\n")

print(y_train)

print("\n")

KNN_Object.fit(X_train, y_train)

# Check the models accuracy in determining whether an input url is malicious or not

print("Testing dataset:")

print("\n")

print(X_test)

print("\n")

print("Expected results from test dataset:")

print("\n")

print(y_test)

print("\n")

ai_model_answers = KNN_Object.determine(X_test)

print("Model proposed answer:")

print("\n")

print(ai_model_answers)

print("\n")

accuracy = np.sum(ai_model_answers == y_test) / len(y_test)

print("Accuracy of model: " + str(accuracy))















	





