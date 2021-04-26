# Time Series projections for Zillow Housing Data For Flatiron School Data Science Immersive (Phase 4 Project)
  
## Overview  
In this project, we examine and perform time series analysis on a data set of housing data from [Zillow Research](https://www.zillow.com/research/data/) to determine whether 1-bedroom or 2-bedrooms homes in San Fransciso would be better for investment on a 1 year time horizon.  

![california_house](images/california_housing.jpg)  
  
## Motivation  
To make informed recommendations to investment advisors, real estate brokers, and houseowners who are looking to invest in a mid-tier 1 or 2-bedroom home in the San Francisco area. This best captures the intent of a couple looking for their first home, or a yuppie looking to buy their first property, and may choose to upgrade and sell for a profit in one year.

## Data  
Our [data set](Zip_zhvi_bdrmcnt_2_uc_sfrcondo_tier_0.33_0.67_sm_sa_mon.csv) comes from the [Research division of Zillow Group](https://www.zillow.com/research/), used with permission. The data represents the typical home values for the zip codes listed for homes falling between the 35th and 65th percentile of home values. General info on the data set is available [here](https://www.zillow.com/research/zhvi-methodology-2019-highlights-26221), and full details are available [here](https://www.zillow.com/research/zhvi-methodology-2019-deep-26226).  
  
### Historical Data Examined  
Typical home values are published on the third Thursday of each month, and gives monthly data from 1996 to present.  

### Target Variable  
We will forecast time series data 12 months in the future for all relevant zip codes (25 in total) in 1-bedroom and 2-bedroom data sets.  We will then compare the top performing zip codes.
  
## Methods  
Our methodology implements the CRISP-DM model for exploratory data analysis, cleaning, modeling, and evaluation.  
We leveraged SARIMAX modeling from [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html) to analysis and forecast the home values. The quality of our modeling was determined with the [AIC value](https://en.wikipedia.org/wiki/Akaike_information_criterion). We also performed statistical analysis via [SciPy Stats](https://docs.scipy.org/doc/scipy/reference/stats.html) to further make inferences on the data.  
Other tools used include Python, NumPy, and Pandas. Visualizations were created with MatPlotLib and Seaborn.  
  
## Conclusion
After running SARIMAX analysis on all zip codes in San Francisco, we found the following zip codes that showed the greatest projected appreciation in value:  
Type of Home    |  Zip Codes                    | Projected </br> Home Value </br> Growth  
:---------------|:------------------------------|-----------------------------:  
1-Bedroom Homes | 94124 </br> 94134 </br> 94132 | 6.03% </br> 5.61% </br> 3.57% 
2-Bedroom Homes | 94121 </br> 94116 </br> 94134 | 4.41% </br> 3.07% </br> 2.74% 

<b>We see that 1-bedroom homes in zip code 94124 shows the greatest project growth in value.</b>  
Perhaps unsurprisingly, this zip code currently has the lowest prices for San Francisco 1-bedroom homes.  According to our model, in a year's time this zip code will be ranked 5th lowest in price out of the 25 zip codes.

## Further Actions  
Using exogenous such as school district data, crime data, presence of parks/nature in vicinity, proximity to hospitals, groceries, entertainment, transportation etc., we can further refine our model as well as assign weights based on what a home buyer is looking to prioritize. We may find that growth in home value may be highly correlated to better education, lower crime rate, etc.   
We would like to further optimize our model by incorporating exogenous data on interest rates, market changes, macroeconomic indicators, and other large scale fluctuations. With more data, we would be able to more accurately forecast home values into the future.  
      
## Index  
- **code/** — directory containing python code files
  - **functions.py** — helper functions
- **crisp_dm_process/** — directory for initial EDA and model notebook files  
  - **SF_Modeling.ipynb** — notebook file that includes thorough initial data exploration, insights, and takeaways  
- **data/** — directory of project data sets as well as model output data
- **images/** — directory containing the following:  
  - image exports of visualizations  
  - additional images for README
- **zillow_housing_final.ipynb** — primary project notebook  
- **Zillow_Housing_Project_JS_WAX.pdf** - presentation slides PDF
- **README.md**  
  
## Bibliography  
1. Dataset Origin:  
  
       Zillow Research Housing Data  
               Zillow Group  
2. Date:    Thursday April 15, 2021
3. Web Source:  https://www.zillow.com/research/data/             
  
<div align="center";>Authors  
  <div align="center";>Jonathan Silverman & Wei Alexander Xin   
    
[Jonathan's GitHub](https://github.com/silvermanjonathan) | [Alexander's GitHub](https://github.com/eggrollofchaos)  
[Jonathan's LinkedIn](https://www.linkedin.com/in/jonathansilverman007) | [Alexander's LinkedIn](https://www.linkedin.com/in/waximus)
