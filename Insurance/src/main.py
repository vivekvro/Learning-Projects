from pydantic import BaseModel,Field,computed_field,field_validator,model_validator
from typing import Annotated,Literal
from fastapi import FastAPI,HTTPException
from fastapi.responses import JSONResponse
import json
import joblib
import pandas as pd

model = joblib.load('Insurance_rf_pipeline.joblib')

def load_data():
    with open('src/data/customer_records.json','r') as f:
        data = json.load(f)
    return data
def save_data(data):
    with open('src/data/customer_records.json','w') as f:
        json.dump(data,f)


class CustomerDetails(BaseModel):
    name: Annotated[str,Field(...,description="Customer Name")]
    age : Annotated[int,Field(...,gt=0,lt=71,description="Customer Age")]
    sex : Annotated[Literal["male", "female"],Field(description="Customer Gender",examples=['female','male'])]
    height : Annotated[float,Field(...,gt=0.3,lt=2.5,description="Customer height in meter")]
    weight: Annotated[float,Field(...,gt=30,lt=131,description="Customer weight in kg")]
    children : Annotated[int,Field(...,lt=6,description="Customer's Children count")]
    smoker: Annotated[Literal["yes", "no"],Field(...,description="does customer Smoke or not, Answer in yes or no",)]
    region: Annotated[Literal['southeast','southwest','northeast','northwest'],Field(...,description="customer's Region",examples=['southeast','southwest','northeast','northwest'])]

    @computed_field
    @property
    def bmi(self)-> float:
        return round((self.weight/(self.height**2)),3)
    @model_validator(mode='after')
    def check_bmi(self):
        if self.bmi < 0 or self.bmi > 60:
            raise ValueError("Invalid Height and Weight input")
        return self


class CustomerInput(BaseModel):
    age : Annotated[int,Field(...,gt=0,lt=71,description="Customer Age")]
    sex : Annotated[Literal["male", "female"],Field(description="Customer Gender",examples=['female','male'])]
    height : Annotated[float,Field(...,gt=0.3,lt=2.5,description="Customer height in meter")]
    weight: Annotated[float,Field(...,gt=30,lt=131,description="Customer weight in kg")]
    children : Annotated[int,Field(...,lt=6,description="Customer's Children count")]
    smoker: Annotated[Literal["yes", "no"],Field(...,description="does customer Smoke or not, Answer in yes or no",)]
    region: Annotated[Literal['southeast','southwest','northeast','northwest'],Field(...,description="customer's Region",examples=['southeast','southwest','northeast','northwest'])]

    @computed_field
    @property
    def bmi(self)-> float:
        return round((self.weight/(self.height**2)),3)
    @model_validator(mode='after')
    def check_bmi(self):
        if self.bmi < 0 or self.bmi > 60:
            raise ValueError("Invalid Height and Weight input")
        return self

app = FastAPI()
@app.get("/customer/records")
def customer_records():
    return load_data()




@app.post("/customer/addcustomer")
def addcustomer(customer_id:str,customerDetails: CustomerDetails):
    data = load_data()


    if customer_id.upper() in data:
        raise HTTPException(status_code=409,detail="Customer Already Exists")
    data[customer_id.upper()] = customerDetails.model_dump(exclude_computed=True)
    save_data(data)

    return JSONResponse(status_code=201, content={"message": "Customer added"})




@app.get("/customer/{customer_id}")
def filter_customer(customer_id:str):
    data = load_data()
    if customer_id.upper() not in data:
        raise HTTPException(status_code=404,detail="Not found")
    return data[customer_id.upper()]



@app.delete("/customer/delete/{customer_id}")
def deleteCustomer(customer_id:str):
    data = load_data()
    if customer_id.upper() not in data:
        raise HTTPException(status_code=404,detail="Already not exist")
    del data[customer_id.upper()]
    save_data(data)

@app.post("/charges")
def predict_charges(data: CustomerInput):
    input_df = pd.DataFrame([{
        "age": data.age,
        "sex": data.sex,
        "bmi": data.bmi,
        "children": data.children,
        "smoker": data.smoker,
        "region": data.region
    }])

    prediction = model.predict(input_df)[0]

    return JSONResponse(
        status_code=200,
        content={"predicted": float(prediction)}
    )
