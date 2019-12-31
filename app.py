from fastai.vision import *
# from fastai.widgets import *

# Get image files 
img_path = './dva.jpg'
print(img_path)
img = open_image(img_path)
img

# # Use CPU instead of GPU
defaults.device = torch.device('cpu')

# # Load pretrained model
model_path = './export.pkl'
learn = load_learner('./')

# # Classify image above
pred_class,pred_idx,outputs = learn.predict(img)
pred_class
print(pred_class)

# @app.route("/classify-url", methods=["GET"])
# async def classify_url(request):
#     bytes = await get_bytes(request.query_params["url"])
#     img = open_image(BytesIO(bytes))
#     _,_,losses = learner.predict(img)
#     return JSONResponse({
#         "predictions": sorted(
#             zip(cat_learner.data.classes, map(float, losses)),
#             key=lambda p: p[1],
#             reverse=True
#         )
#     })