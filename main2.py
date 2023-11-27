import pandas as pd
import turicreate as tc

user_id = 'id'
item_name = "part_name"
n_rec = 3
data_dummy = pd.read_csv("brandpart.csv")

purchase_id= 5
make  = input("Enter Make:- ")
model = input("Enter model:- ")
product_name = input("Enter part Name:- ")
purcase_count = int(input("Enter item Quantiy:- "))

prod = str(make)+"_"+str(model)+"_"+str(product_name)

data_dummy.loc[len(data_dummy.index)] = [purchase_id,prod, purcase_count]

final_model = tc.item_similarity_recommender.create(tc.SFrame(data_dummy),
                                    user_id=user_id,item_id=item_name,
                                    target='purchase_count', similarity_type='cosine',verbose=0)

def create_output(model, user_to_recommend, n_rec):
    recomendation = model.recommend(users=user_to_recommend, k=n_rec)
    return recomendation

user_to_recommend_item =[purchase_id]
df_output = create_output(final_model,user_to_recommend_item , n_rec)

out= df_output["part_name"]
for i in range(n_rec):
    a= out[i].split("_")[0]
    b= out[i].split("_")[1]
    c= out[i].split("_")[2]

    print(f"Make:-{a}, and Model:-{b}, and Item :- {c}")