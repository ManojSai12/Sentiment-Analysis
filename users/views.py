from django.shortcuts import render, HttpResponse
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel
from textblob import TextBlob
import re

# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account has not been activated by Admin.')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})


def user_view_dataset(request):
    from django.conf import settings
    import pandas as pd
    path = settings.MEDIA_ROOT + "\\" + "amazon_reviews.csv"
    df = pd.read_csv(path)

    df = df.head(100).to_html
    return render(request, 'users/dataset_view.html', {'data': df})


def assign_sentiment(rating):
    if float(rating) >= 4:
        return "Positive"
    else:
        return "Negative"


def user_view_sentiment(request):
    from django.conf import settings
    import pandas as pd
    path = settings.MEDIA_ROOT + "\\" + "amazon_reviews.csv"
    df = pd.read_csv(path, dtype='unicode')
    df.columns = ['id', 'name', 'asins', 'brand', 'categories', 'keys', 'manufacturer', 'date', 'dateAdded', 'dateSeen',
                  'isPurchase', 'isRecommended', 'reviewsId', 'numHelpful', 'rating', 'sourceURLs', 'reviewText',
                  'reviewTitle', 'city', 'userProvince', 'username']
    df = df.head(500)
    print(df.reviewsId.dtype)
    df.reviewsId.fillna(0.0)
    print(df.head(10))
    df = df.drop('keys', 1)
    df.drop('sourceURLs', 1, inplace=True)
    df.drop(['dateAdded', 'dateSeen'], 1, inplace=True)
    # df.head()
    df.isPurchase.fillna(False, inplace=True)
    df.reviewsId.fillna("", inplace=True)
    df.city.fillna("", inplace=True)
    df.userProvince.fillna("", inplace=True)
    # df.head()
    print(df.describe(include='object'))
    df.dropna(subset=['name'], inplace=True)
    df.describe(include='object')
    df['name'].value_counts()
    sdf = df[['rating', 'reviewText']]
    sdf.head(2)
    sdf['sentiment'] = sdf['rating'].apply(assign_sentiment)
    sdf.drop('rating', inplace=True, axis=1)
    sdf = sdf.to_html
    return render(request, 'users/user_view_sentiment.html', {'data': sdf})


def user_classifiers(request):
    from .utility import amazon_process
    result = amazon_process.start_classification_analysis()
    return render(request, 'users/user_classification_result.html', result)

def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w+:\/\/\s+)"," ", tweet).split())


def user_predictions(request):
    if request.method=='POST':
        tweet = request.POST.get('tweet')
        analysis = TextBlob(clean_tweet(tweet))
        score = analysis.sentiment.polarity
        msg = ''
        if score >0:
            msg = 'Positive'
        elif score==0:
            msg = 'Neutral'
        else:
            msg = 'Negative'
        return render(request, 'users/user_predict_form.html',{'msg': msg, 'tweet': tweet})            

    else:
        return render(request, 'users/user_predict_form.html',{})
