from flask import Flask, render_template, request, redirect, url_for, make_response
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc
from sqlalchemy.sql.expression import case
import hashlib, math, re, time
import sqlite3
import numpy as np
import pandas as pd
from surprise import SVD, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate

DATABASE_NAME = 'tavsiye.db'

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + DATABASE_NAME
db = SQLAlchemy(app)

class RecommendationSystem:
    def __init__(self):        
        app.add_url_rule('/', 'index', self.index)
        app.add_url_rule('/detail/rating', 'rateGlass', self.rateGlass, methods = ["POST"])
        app.add_url_rule("/register", 'register', self.register, methods = ["POST","GET"])
        app.add_url_rule("/login", 'login', self.login,  methods = ["POST","GET"])
        app.add_url_rule("/products", 'showAllProducts', self.showAllProducts, methods = ["POST", "GET"])
        app.add_url_rule("/detail/<string:id>", 'showDetail', self.showDetail, methods = ["GET"])

    def index(self):
        if 'username' not in request.cookies:
            return render_template("index.html")
        else:
            return redirect(url_for("showAllProducts"))
          
    def login(self):
        if 'username' not in request.cookies:
            if request.method == 'POST':
                username = request.form.get("username")
                password = hashlib.md5(request.form.get("password").encode('utf-8')).hexdigest()
                
                if(db.session.query(User.userID).filter_by(username=username, password=password).scalar() is None):
                    error = "Kullanıcı adı veya şifreniz yanlış!"
                    return render_template("login.html", error = error)
                else:
                    user = User.query.filter_by(username=username, password=password).first()
                    glasses = Glasses.query.all()
                    
                    response = make_response(redirect(url_for("showAllProducts")))
                    response.set_cookie('username', username)
                    return response
            else:
                return render_template("login.html")
        else:
            return redirect(url_for("showAllProducts"))
            
    def register(self):
        if request.method == 'POST':
            username = request.form.get("username")
            password = hashlib.md5(request.form.get("password").encode('utf-8')).hexdigest()
            gender = request.form.get("gender")
            faceType = request.form.get("faceType")
            
            if(db.session.query(User.userID).filter_by(username=username, password=password).scalar() is not None):
                error = "Böyle bir kullanıcı mevcut."
                return render_template("index.html", error = error)
            else:
                newUser = User(username = username, password = password, gender = gender, faceType = faceType)
                
                db.session.add(newUser)
                db.session.commit()
                return render_template("login.html", newUser = newUser)
        else:
            return redirect(url_for("index"))    

    def rateGlass(self):
        rating = float(request.form['rating'])
        glassID = int(request.form['glassID'])
        
        username = request.cookies.get('username')  
        userID = User.query.filter_by(username = username).first().userID
        
        newRating = Ratings(userID = userID, glassID = glassID, rating = rating, relativeRating = rating)
        
        if(db.session.query(Ratings.userID, Ratings.glassID).filter_by(userID = userID, glassID = glassID).scalar() is not None):
            newRating = Ratings.query.filter_by(userID = userID, glassID = glassID).first()
            newRating.relativeRating = newRating.relativeRating + (rating - newRating.rating)
            newRating.rating = rating
            db.session.commit()
        else:
            db.session.add(newRating)
            db.session.commit()
        return ("nothing")  

    def showAllProducts(self):
        if 'username' in request.cookies:
            username = request.cookies.get('username')
            userID = User.query.filter_by(username = username).first().userID
            page = request.args.get('page', 1, type=int)

            genders = None
            frames = None
            brands = None
            predictAlgorithm = None
            similarityMeasure = None
            isUserBased = None

            # Get detail session
            if 'detailSession' in request.cookies and 'glassID' in request.cookies:
                detailSession = int(time.time() - float(request.cookies.get('detailSession')))
                glassID = request.cookies.get('glassID')
                
                ratingRate = 0
                
                if detailSession > 0 and detailSession <= 30:
                    ratingRate = 0.25
                elif detailSession > 30 and detailSession <= 60:
                    ratingRate = 0.50
                elif detailSession > 60 and detailSession <= 120:
                    ratingRate = 0.75
                else:
                    ratingRate = 1.00
                
                # Adjust the relative rating
                newRating = Ratings(userID = userID, glassID = glassID, rating = 0, relativeRating = ratingRate)
        
                if(db.session.query(Ratings.userID, Ratings.glassID).filter_by(userID = userID, glassID = glassID).scalar() is not None):
                    newRating = Ratings.query.filter_by(userID = userID, glassID = glassID).first()
                    newRatingTmp = newRating.relativeRating + ratingRate
                    if newRatingTmp > 5:
                        newRatingTmp = 5
                    newRating.relativeRating = newRatingTmp
                    db.session.commit()
                else:
                    db.session.add(newRating)
                    db.session.commit()
                
            # Apply filters
            if request.method == 'GET':
                genders = request.args.get("gender")
                frames = request.args.get("frame")
                brands = request.args.get("brand")
                predictAlgorithm = request.args.get("predictAlgorithm")           
            else:
                try:
                    genders = request.form['gender']
                except:
                    pass
                try:
                    frames = request.form['frame']
                except:
                    pass
                try:
                    brands = request.form['brand']
                except:
                    pass
                try:
                    predictAlgorithm = request.form['predictAlgorithm']
                except:
                    pass
                try:
                    similarityMeasure = request.form['similarityMeasure']
                except:
                    pass
                try:
                    isUserBased = request.form['isUserBased']
                except:
                    pass

            glasses = Glasses.query.filter(Glasses.glassID > 0)
     
            # Apply predict algorithm filter
            if predictAlgorithm is None:
                predictAlgorithm = 9 # Default prediction algorithm is baselineonly
            
            # Apply similarity algorithm filter
            if similarityMeasure is None:
                similarityMeasure = 1 # Default similarity measure is cosine

            # Apply user based filter
            if isUserBased is None:
                isUserBased = "Yes" # Default is user based

            # Show the recommendations
            recommendations = []
            rmseValue = 0
            
            if(db.session.query(Ratings.userID).filter_by(userID = userID).count() == 0):
                userIdentity = User.query.filter_by(userID = userID).first()
                recommendedFrames = []
                if (userIdentity.faceType == 'Kalp'):
                    recommendedFrames = ['Çekik','Köşeli']
                elif (userIdentity.faceType == 'Kare'):
                    recommendedFrames = ['Çekik','Oval', 'Yuvarlak']
                elif (userIdentity.faceType == 'Oval'):
                    recommendedFrames = ['Çekik','Damla','Kelebek', 'Köşeli','Oval','Yuvarlak']
                elif (userIdentity.faceType == 'Yuvarlak'):
                    recommendedFrames = ['Çekik','Köşeli','Oval']
                recommendations = Glasses.query.filter((Glasses.gender == userIdentity.gender) & (Glasses.frame.in_(recommendedFrames))).all()
            else:
                # Advanced recommendation
                userIdentity = User.query.filter_by(userID = userID).first()
                recommendationsIDs = self.getRecommendations(userID, int(predictAlgorithm), int(similarityMeasure), str(isUserBased))
                ordering = case({glassID: index for index, glassID in enumerate(recommendationsIDs)},value=Glasses.glassID)
                recommendations = Glasses.query.filter(Glasses.glassID.in_(recommendationsIDs) & ((Glasses.gender == userIdentity.gender) | (Glasses.gender == 'Unisex'))).order_by(ordering).limit(10).all()
                rmseValue = self.calculateRMSE(int(predictAlgorithm), int(similarityMeasure), str(isUserBased))
                 
            # Apply gender filter
            if genders is not None:
                genders = genders.split(',')
                if (db.session.query(Glasses.gender).filter(Glasses.gender.in_(genders)).count() == 0):
                    return redirect(url_for("showAllProducts"))
                else:
                    glasses = glasses.filter(Glasses.gender.in_(genders))
            
            # Apply frame filter
            if frames is not None:
                frames = frames.split(',')
                if (db.session.query(Glasses.frame).filter(Glasses.frame.in_(frames)).count() == 0):
                    return redirect(url_for("showAllProducts"))
                else:
                    glasses = glasses.filter(Glasses.frame.in_(frames))
            
            # Apply brand filter
            if brands is not None:
                brands = brands.split(',')
                if (db.session.query(Glasses.brand).filter(Glasses.brand.in_(brands)).count() == 0):
                    return redirect(url_for("showAllProducts"))
                else:
                    glasses = glasses.filter(Glasses.brand.in_(brands))
            
            glasses = glasses.paginate(page = page, per_page=24)
                  
            # Get the filter options
            genderList = db.session.query(Glasses.gender).distinct()
            brandList = db.session.query(Glasses.brand).distinct()
            frameList = db.session.query(Glasses.frame).distinct()
            
            genderListFinal = []
            brandListFinal = []
            frameListFinal = []
            
            for gender in genderList:
                genderListFinal.append(re.sub("'|\(|\)|,", "", str(gender)))
            for brand in brandList:
                brandListFinal.append(re.sub("'|\(|\)|,", "", str(brand)))
            for frame in frameList:
                frameListFinal.append(re.sub("\*|'|\(|\)|,|[0-9]", "", str(frame)))
                frameListFinal = ' '.join(frameListFinal).split() 
            
            # Create the responses for the filters
            if genders is not None:
                if len(genders) == 1:
                    if frames is not None:
                        if len(frames) == 1:
                            response = make_response(render_template("products.html", rmseValue = rmseValue, recommendations = recommendations, glasses = glasses, gender = genders[0], frame = frames[0], genderList = genderListFinal, brandList = brandListFinal, frameList = frameListFinal, totalPage = math.ceil(glasses.total / glasses.per_page)))
                            response.delete_cookie('detailSession')
                            response.delete_cookie('glassID')
                            return response
                        else:
                            response = make_response(render_template("products.html", rmseValue = rmseValue, glasses = glasses, recommendations = recommendations, gender = genders[0], genderList = genderListFinal, brandList = brandListFinal, frameList = frameListFinal, totalPage = math.ceil(glasses.total / glasses.per_page)))
                            response.delete_cookie('detailSession')
                            response.delete_cookie('glassID')
                            return response
                    else:
                        response = make_response(render_template("products.html", rmseValue = rmseValue, glasses = glasses, recommendations = recommendations, gender = genders[0], genderList = genderListFinal, brandList = brandListFinal, frameList = frameListFinal, totalPage = math.ceil(glasses.total / glasses.per_page)))
                        response.delete_cookie('detailSession')
                        response.delete_cookie('glassID')
                        return response
                else:
                    response = make_response(render_template("products.html", rmseValue = rmseValue, glasses = glasses, recommendations = recommendations, genderList = genderListFinal, brandList = brandListFinal, frameList = frameListFinal, totalPage = math.ceil(glasses.total / glasses.per_page)))
                    response.delete_cookie('detailSession')
                    response.delete_cookie('glassID')
                    return response
            else:
                response = make_response(render_template("products.html", rmseValue = rmseValue, glasses = glasses, recommendations = recommendations, genderList = genderListFinal, brandList = brandListFinal, frameList = frameListFinal, totalPage = math.ceil(glasses.total / glasses.per_page)))
                response.delete_cookie('detailSession')
                response.delete_cookie('glassID')
                return response
        else:
            return redirect(url_for("index"))

    def showDetail(self, id):
        if 'username' in request.cookies:
            if (db.session.query(Glasses.glassID).filter_by(glassID=id).scalar() is None):
                return redirect(url_for("index"))
                    
            username = request.cookies.get('username')  
            userID = User.query.filter_by(username = username).first().userID
            
            product = Glasses.query.filter_by(glassID=id).first()
            
            rating = -1.0
            
            if (db.session.query(Ratings.userID, Ratings.glassID).filter_by(userID = userID, glassID = id).scalar() is not None):
                rating = Ratings.query.filter_by(userID = userID, glassID = id).first().rating
            
            response = make_response(render_template("detail.html", product = product, rating = rating))
            response.set_cookie('detailSession', str(time.time()))
            response.set_cookie('glassID', id)
            return response
        else:
            return redirect(url_for("index"))


    # Advanced Recommendation
    def getRecommendations(self, IDUser, method = 9, similarityMeasure = 1, isUserBased = "Yes"):
        conn = sqlite3.connect(DATABASE_NAME)
        df = pd.read_sql_query("SELECT userID, glassID, relativeRating FROM ratings", conn)
        
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['userID', 'glassID', 'relativeRating']], reader)
    
        trainset = data.build_full_trainset()

        isUserBased = True if (isUserBased == "Yes") else False
        if similarityMeasure == 1:
            similarityMeasure = "cosine"
        elif similarityMeasure == 2:
            similarityMeasure = "pearson"
        else:
            similarityMeasure = "pearson_baseline"

        sim_options = {'name': similarityMeasure, 'user_based': isUserBased}

        if method == 1:
            algo = SVD()
        elif method == 2:
            algo = SlopeOne()
        elif method == 3:
            algo = NMF()
        elif method == 4:
            algo = NormalPredictor()
        elif method == 5:
            algo = KNNBaseline(sim_options=sim_options)
        elif method == 6:
            algo = KNNBasic(sim_options=sim_options)
        elif method == 7:
            algo = KNNWithMeans(sim_options=sim_options)
        elif method == 8:
            algo = KNNWithZScore(sim_options=sim_options)
        elif method == 9:
            algo = BaselineOnly()
        else:
            algo = CoClustering()

        algo.fit(trainset)

        predictions = pd.DataFrame(columns=['glassID', 'estimatedRating'])
        
        totalGlass = df['glassID'].max()
        
        glassPivot = df.pivot_table(index='glassID', columns='userID', values='relativeRating') 
            
        for iid in range(1, totalGlass + 1):
            isNan = True
            
            try:
                isNan = pd.isna(glassPivot.loc[iid, IDUser])
            except:
                continue
            
            if isNan:
                prediction = algo.predict(IDUser, iid, verbose=False)
                predictions = predictions.append(pd.DataFrame([[iid, prediction[3]]], columns=predictions.columns))
                    
        
        predictions = predictions.sort_values('estimatedRating', ascending=False)
        recommendationList = [item for item in predictions[predictions['estimatedRating'] > 3]['glassID'].head(50).tolist()]    

        conn.close()

        return recommendationList


    def calculateRMSE(self, method = 9, similarityMeasure = 1, isUserBased = "Yes"):
        conn = sqlite3.connect(DATABASE_NAME)
        df = pd.read_sql_query("SELECT userID, glassID, relativeRating FROM ratings", conn)
        
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['userID', 'glassID', 'relativeRating']], reader)
    
        trainset, testset = train_test_split(data, test_size=.20)
                        
        isUserBased = True if (isUserBased == "Yes") else False
        if similarityMeasure == 1:
            similarityMeasure = "cosine"
        elif similarityMeasure == 2:
            similarityMeasure = "pearson"
        else:
            similarityMeasure = "pearson_baseline"

        sim_options = {'name': similarityMeasure, 'user_based': isUserBased}

        if method == 1:
            algo = SVD()
        elif method == 2:
            algo = SlopeOne()
        elif method == 3:
            algo = NMF()
        elif method == 4:
            algo = NormalPredictor()
        elif method == 5:
            algo = KNNBaseline(sim_options=sim_options)
        elif method == 6:
            algo = KNNBasic(sim_options=sim_options)
        elif method == 7:
            algo = KNNWithMeans(sim_options=sim_options)
        elif method == 8:
            algo = KNNWithZScore(sim_options=sim_options)
        elif method == 9:
            algo = BaselineOnly()
        else:
            algo = CoClustering()

        algo.fit(trainset)
        predictions = algo.test(testset)

        conn.close()

        #cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=True)
        return round(accuracy.rmse(predictions, verbose=False), 4)

class User(db.Model):
    userID = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(80), unique=False, nullable=False)
    gender = db.Column(db.String(10), unique=False, nullable=False)
    faceType = db.Column(db.String(15), unique=False, nullable=False)

class Glasses(db.Model):
    glassID = db.Column(db.Integer, primary_key=True)
    brand = db.Column(db.String(30), unique=False, nullable=False)
    description = db.Column(db.Text, unique=False, nullable=False)
    properties = db.Column(db.Text, unique=False, nullable=False)
    model = db.Column(db.String(80), unique=False, nullable=False)
    image = db.Column(db.String(130), unique=False, nullable=False)
    image2 = db.Column(db.String(130), unique=False, nullable=False)
    brandImg = db.Column(db.String(130), unique=False, nullable=False)
    frame = db.Column(db.String(30), unique=False, nullable=False)
    gender = db.Column(db.String(10), unique=False, nullable=False)
    
class Ratings(db.Model):
    userID = db.Column(db.Integer, primary_key=True)
    glassID = db.Column(db.Integer, primary_key=True)
    rating = db.Column(db.Float(5), unique=False, nullable=False)
    relativeRating = db.Column(db.Float(5), unique=False, nullable=False)
    
    
if __name__ == "__main__":
    recommendationSystem = RecommendationSystem()
    app.run(debug=True)

