import pandas as pd
import turicreate as tc

user_id = 'id'
item_name = "item_name"
n_rec = 3

data_dummy = pd.read_csv("my_data.csv")


purchase_id= 5
part_name = input("Enter part Name:- ")


data_dummy.loc[len(data_dummy.index)] = [purchase_id,part_name, 1]


final_model = tc.item_similarity_recommender.create(tc.SFrame(data_dummy),
                                    user_id=user_id,item_id=item_name,
                                    target='purchase_count', similarity_type='cosine',verbose=0)

def create_output(model, user_to_recommend, n_rec):
    recomendation = model.recommend(users=user_to_recommend, k=n_rec)
    return recomendation

user_to_recommend_item =[purchase_id]
df_output = create_output(final_model,user_to_recommend_item , n_rec)


for i in range(n_rec):

    print(f"{i+1} "+df_output["item_name"][i])