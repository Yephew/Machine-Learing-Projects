# Given yelp business cases in a csv file, return custom sentiment for each business.
# NLP text process.

#business = {} # Save all the business with their score
sentiment = []
data = []
with open('/Users/yephewzhang/Downloads/reviews_data.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['business_id'] != '':
            data.append({'business_id': row['business_id'],
                        'business_review_count': row['business_review_count'],
                        'review_id': row['review_id'],
                        'reviewer_average_stars': row['reviewer_average_stars'],
                        'reviewer_review_count': row['reviewer_review_count'],
                        'reviewer_useful': row['reviewer_useful'],
                        'text': row['text'],
                        })
                        review = TextBlob(row['text']).sentiment.polarity
                        sentiment.append(review)
                        if     -1 <= review < -0.6:data[-1]['star'] = 1
                        elif -0.6 <= review < -0.2:data[-1]['star'] = 2
                        elif -0.2 <= review <  0.2:data[-1]['star'] = 3
                        elif  0.2 <= review <  0.6:data[-1]['star'] = 4
                        else:data[-1]['star'] = 5

