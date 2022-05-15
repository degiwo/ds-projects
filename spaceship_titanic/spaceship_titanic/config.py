# Store all configurations like paths, ...

PATH_DATA_FOLDER = "./spaceship_titanic/data/"
PATH_MODEL_FOLDER = "./spaceship_titanic/model/"

COLUMN_TARGET = "Transported"
COLUMN_TOTAL_BILL = [
    # columns to sum for new column TotalBill
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck"
]
COLUMN_IMPUTE = [
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck"
]
COLUMN_ONEHOT = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "Age",
    "VIP"
]
