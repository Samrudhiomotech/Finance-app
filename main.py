import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Set page configuration
st.set_page_config(
    page_title="Savings Behavior & Financial Resilience Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.3rem;
        margin-top: 1.5rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0px;
        font-weight: bold;
        text-align: center;
    }
    .feature-importance {
        background-color: #f0f0f0;
        padding: 15px;
        border-radius: 10px;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0px;
        border-left: 5px solid #1f77b4;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .scheme-card {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0px;
        border-left: 5px solid #1f77b4;
    }
    .recommended-scheme {
        background-color: #e8f5e8;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0px;
        border-left: 5px solid #28a745;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .priority-high {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .priority-medium {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
    }
    .priority-low {
        background-color: #f8f9fa;
        border-left: 5px solid #6c757d;
    }
    .stButton button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    .recommendation-score {
        font-size: 1.2em;
        font-weight: bold;
        color: #28a745;
    }
    .max-limit {
        font-size: 0.8em;
        color: #666;
        margin-top: -10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load the model with error handling
@st.cache_resource
def load_model():
    try:
        # Suppress sklearn warnings during model loading
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            artifacts = joblib.load('savings_model.pkl')
        return artifacts
    except FileNotFoundError:
        st.error("Model file not found. Please make sure 'savings_model.pkl' is in the same directory.")
        return None
    except Exception as e:
        st.warning(f"Model loaded with compatibility warnings. The dashboard will work with estimated calculations.")
        return None

# Function to estimate savings amount based on features
def estimate_savings_amount(input_data):
    """
    Estimate the actual savings amount based on input features
    This model combines financial, demographic, and regional factors
    """
    # Base savings calculation (percentage of income)
    base_savings_rate = 0.15  # Base savings rate
    
    # Adjust savings rate based on financial factors
    debt_impact = max(0, 1 - (input_data['Debt_to_Income'] * 0.7))  # Higher debt reduces savings
    expense_impact = max(0, 1 - (input_data['Expense_Ratio'] * 0.4))  # Higher expenses reduce savings
    emergency_impact = 1 + (input_data['Emergency_Ratio'] * 0.3)  # Emergency fund increases savings
    
    # Adjust based on employment status
    if input_data['Employment_Status'] == 'Employed':
        employment_impact = 1.2
    elif input_data['Employment_Status'] == 'Self-Employed':
        employment_impact = 1.1
    else:  # Unemployed or other
        employment_impact = 0.8
    
    # Adjust based on education
    if input_data['Education'] == 'Post Graduate':
        education_impact = 1.3
    elif input_data['Education'] == 'Graduate':
        education_impact = 1.2
    elif input_data['Education'] == 'High School':
        education_impact = 1.0
    else:
        education_impact = 0.9
    
    # Regional factors (CPI and poverty rate)
    cpi_impact = 1 - ((input_data['CPI_Index'] - 100) / 100 * 0.2)  # Higher CPI reduces real savings
    poverty_impact = 1 - (input_data['Poverty_Rate'] * 0.5)  # Higher poverty rate reduces savings
    
    # MGNREGA impact (more days = more stable income)
    mgnrega_impact = 1 + (min(input_data['Avg_MGNREGA_Days'], 60) / 60 * 0.2)
    
    # Calculate final savings rate
    savings_rate = base_savings_rate * debt_impact * expense_impact * emergency_impact * \
                  employment_impact * education_impact * cpi_impact * poverty_impact * mgnrega_impact
    
    # Calculate estimated savings amount
    estimated_savings = input_data['Income'] * savings_rate
    
    # Ensure savings is not negative
    estimated_savings = max(0, estimated_savings)
    
    return estimated_savings, savings_rate

# Function to calculate financial resilience score
def calculate_resilience_score(input_data, savings_amount):
    """
    Calculate a financial resilience score (0-100) based on household characteristics
    """
    score = 50  # Base score
    
    # Income factors (max +15)
    if input_data['Income'] > 50000:
        score += 10
    elif input_data['Income'] > 30000:
        score += 5
    
    # Debt factors (max -20)
    if input_data['Debt_to_Income'] > 0.7:
        score -= 15
    elif input_data['Debt_to_Income'] > 0.4:
        score -= 7
    
    # Emergency fund factors (max +15)
    if input_data['Emergency_Ratio'] > 0.3:
        score += 10
    elif input_data['Emergency_Ratio'] > 0.1:
        score += 5
    
    # Employment factors (max +10)
    if input_data['Employment_Status'] == 'Employed':
        score += 7
    elif input_data['Employment_Status'] == 'Self-Employed':
        score += 4
    
    # Savings factors (max +20)
    savings_ratio = savings_amount / input_data['Income'] if input_data['Income'] > 0 else 0
    if savings_ratio > 0.2:
        score += 15
    elif savings_ratio > 0.1:
        score += 8
    elif savings_ratio > 0.05:
        score += 3
    
    # Regional factors (max +10)
    if input_data['Poverty_Rate'] < 0.15:
        score += 7
    elif input_data['Poverty_Rate'] < 0.25:
        score += 3
    
    # Ensure score is between 0 and 100
    return max(0, min(100, score))

# Function to determine user profile based on input data
def determine_user_profile(input_data, savings_amount, resilience_score):
    """
    Determine user profile for scheme recommendations
    """
    # Check if income is zero to avoid division by zero
    income = input_data['Income'] if input_data['Income'] > 0 else 1
    
    profile = {
        'has_bank_account': True,  # Assume true for this analysis
        'age_group': 'adult',  # Could be enhanced with actual age input
        'has_girl_child': False,  # Could be enhanced with family info
        'income_level': 'low' if input_data['Income'] < 20000 else 'medium' if input_data['Income'] < 40000 else 'high',
        'employment_type': 'organized' if input_data['Employment_Status'] == 'Employed' else 'unorganized',
        'savings_behavior': 'good' if savings_amount / income > 0.15 else 'moderate' if savings_amount / income > 0.05 else 'poor',
        'financial_stability': 'stable' if resilience_score > 70 else 'moderate' if resilience_score > 50 else 'unstable',
        'debt_burden': 'high' if input_data['Debt_to_Income'] > 0.6 else 'moderate' if input_data['Debt_to_Income'] > 0.3 else 'low',
        'emergency_preparedness': 'good' if input_data['Emergency_Ratio'] > 0.2 else 'moderate' if input_data['Emergency_Ratio'] > 0.1 else 'poor'
    }
    return profile

# Enhanced scheme recommendation function
def recommend_schemes(user_profile, input_data, savings_amount):
    """
    Recommend government schemes based on user profile and financial data
    """
    recommendations = []
    
    # Define scheme details
    schemes = {
        'PMJDY': {
            'name': 'Pradhan Mantri Jan Dhan Yojana',
            'type': 'Banking',
            'benefits': ['Zero-balance account', 'Free RuPay card', 'Overdraft facility', 'Insurance cover'],
            'eligibility': 'All citizens',
            'priority_factors': ['no_bank_account', 'low_income', 'unorganized_sector'],
            'website': 'https://pmjdy.gov.in'
        },
        'APY': {
            'name': 'Atal Pension Yojana',
            'type': 'Pension',
            'benefits': ['Fixed pension ₹1000-₹5000/month', 'Government co-contribution', 'Tax benefits'],
            'eligibility': 'Age 18-40, Unorganized sector',
            'priority_factors': ['unorganized_sector', 'no_pension', 'young_adult'],
            'website': 'https://npstrust.org.in/content/atal-pension-yojana'
        },
        'PPF': {
            'name': 'Public Provident Fund',
            'type': 'Long-term Savings',
            'benefits': ['7.1% tax-free returns', '15-year tenure', 'EEE tax benefit'],
            'eligibility': 'All Indian residents',
            'priority_factors': ['stable_income', 'tax_saving_needed', 'long_term_goals'],
            'website': 'https://www.indiapost.gov.in'
        },
        'SSY': {
            'name': 'Sukanya Samriddhi Yojana',
            'type': 'Girl Child Savings',
            'benefits': ['8.2% tax-free returns', 'For girl child education/marriage', 'Tax benefits'],
            'eligibility': 'Girl child below 10 years',
            'priority_factors': ['girl_child', 'education_planning', 'tax_saving_needed'],
            'website': 'https://www.indiapost.gov.in'
        },
        'PMSBY': {
            'name': 'Pradhan Mantri Suraksha Bima Yojana',
            'type': 'Insurance',
            'benefits': ['₹2 lakh accidental death/disability cover', '₹20/year premium', 'Auto-debit'],
            'eligibility': 'Age 18-70, Bank account holder',
            'priority_factors': ['no_insurance', 'daily_wage_earner', 'risk_prone_job'],
            'website': 'https://jansuraksha.gov.in'
        },
        'PMJJBY': {
            'name': 'Pradhan Mantri Jeevan Jyoti Bima Yojana',
            'type': 'Life Insurance',
            'benefits': ['₹2 lakh life cover', '₹330/year premium', 'Renewable annually'],
            'eligibility': 'Age 18-50, Bank account holder',
            'priority_factors': ['family_breadwinner', 'no_life_insurance', 'moderate_income'],
            'website': 'https://jansuraksha.gov.in'
        }
    }
    
    # Calculate recommendation scores for each scheme
    for scheme_id, scheme in schemes.items():
        score = 0
        reasons = []
        
        # Base scoring logic
        if scheme_id == 'PMJDY':
            if not user_profile['has_bank_account']:
                score += 40
                reasons.append("No existing bank account")
            if user_profile['income_level'] == 'low':
                score += 25
                reasons.append("Low income - benefits from zero-balance account")
            if user_profile['employment_type'] == 'unorganized':
                score += 20
                reasons.append("Unorganized sector worker")
            if input_data['Emergency_Ratio'] < 0.1:
                score += 15
                reasons.append("Limited emergency fund - overdraft facility helpful")
        
        elif scheme_id == 'APY':
            if user_profile['employment_type'] == 'unorganized':
                score += 35
                reasons.append("Unorganized sector - no employer pension")
            if 18 <= 40:  # Assuming age range, could be enhanced
                score += 25
                reasons.append("Suitable age group for pension planning")
            if user_profile['financial_stability'] in ['stable', 'moderate']:
                score += 20
                reasons.append("Financially stable enough for regular contributions")
            if user_profile['income_level'] in ['low', 'medium']:
                score += 15
                reasons.append("Government co-contribution available")
        
        elif scheme_id == 'PPF':
            if user_profile['income_level'] in ['medium', 'high']:
                score += 30
                reasons.append("Sufficient income for regular investments")
            if user_profile['savings_behavior'] in ['good', 'moderate']:
                score += 25
                reasons.append("Good savings behavior for long-term commitment")
            if input_data['Income'] > 25000:
                score += 20
                reasons.append("Tax benefits valuable at current income level")
            if user_profile['employment_type'] == 'organized':
                score += 15
                reasons.append("Stable employment supports 15-year commitment")
        
        elif scheme_id == 'SSY':
            if user_profile['has_girl_child']:
                score += 50
                reasons.append("Has girl child - primary eligibility met")
            if user_profile['savings_behavior'] in ['good', 'moderate']:
                score += 25
                reasons.append("Good savings behavior for child's future")
            if user_profile['income_level'] in ['medium', 'high']:
                score += 15
                reasons.append("Sufficient income for regular deposits")
        
        elif scheme_id == 'PMSBY':
            if input_data['Designation'] == 'Daily Wage':
                score += 30
                reasons.append("High-risk occupation - accident insurance important")
            if user_profile['emergency_preparedness'] == 'poor':
                score += 25
                reasons.append("Limited emergency fund - insurance provides safety net")
            if user_profile['income_level'] == 'low':
                score += 20
                reasons.append("Low-cost insurance suitable for income level")
            if user_profile['has_bank_account']:
                score += 15
                reasons.append("Auto-debit facility available")
        
        elif scheme_id == 'PMJJBY':
            if user_profile['employment_type'] == 'organized' or input_data['Income'] > 15000:
                score += 30
                reasons.append("Primary breadwinner - life insurance essential")
            if user_profile['debt_burden'] in ['high', 'moderate']:
                score += 25
                reasons.append("Debt obligations - family protection needed")
            if user_profile['income_level'] in ['low', 'medium']:
                score += 20
                reasons.append("Affordable premium for income level")
        
        # Add scheme to recommendations if score > 0
        if score > 0:
            priority = 'High' if score >= 60 else 'Medium' if score >= 30 else 'Low'
            recommendations.append({
                'scheme_id': scheme_id,
                'scheme': scheme,
                'score': score,
                'priority': priority,
                'reasons': reasons,
                'estimated_benefit': calculate_estimated_benefit(scheme_id, input_data, savings_amount)
            })
    
    # Sort by score (highest first)
    recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)
    
    return recommendations

def calculate_estimated_benefit(scheme_id, input_data, savings_amount):
    """
    Calculate estimated financial benefit for each scheme
    """
    benefits = {}
    
    if scheme_id == 'PMJDY':
        benefits = {
            'overdraft_limit': min(10000, input_data['Income'] * 0.5) if input_data['Income'] > 0 else 0,
            'insurance_cover': 130000,
            'annual_cost': 0
        }
    elif scheme_id == 'APY':
        monthly_contribution = min(input_data['Income'] * 0.05, 1000) if input_data['Income'] > 0 else 0  # 5% of income or ₹1000, whichever is less
        pension_amount = min(monthly_contribution * 10, 5000)  # Simplified calculation
        benefits = {
            'monthly_contribution': monthly_contribution,
            'monthly_pension_at_60': pension_amount,
            'annual_cost': monthly_contribution * 12
        }
    elif scheme_id == 'PPF':
        max_annual_investment = min(150000, savings_amount * 12 * 0.8)  # 80% of annual savings or max limit
        maturity_value = max_annual_investment * 15 * 1.071 ** 15  # Simplified compound calculation
        benefits = {
            'max_annual_investment': max_annual_investment,
            'estimated_maturity_value': maturity_value,
            'annual_tax_savings': min(max_annual_investment, 150000) * 0.2  # 20% tax saving
        }
    elif scheme_id == 'SSY':
        max_annual_investment = min(150000, savings_amount * 12 * 0.6)  # 60% of annual savings
        maturity_value = max_annual_investment * 15 * 1.082 ** 21  # 21 years compound calculation
        benefits = {
            'max_annual_investment': max_annual_investment,
            'estimated_maturity_value': maturity_value,
            'annual_tax_savings': min(max_annual_investment, 150000) * 0.2
        }
    elif scheme_id == 'PMSBY':
        benefits = {
            'accidental_cover': 200000,
            'partial_disability_cover': 100000,
            'annual_premium': 20
        }
    elif scheme_id == 'PMJJBY':
        benefits = {
            'life_cover': 200000,
            'annual_premium': 330
        }
    
    return benefits

# Function to generate insights
def generate_insights(input_data, savings_amount, resilience_score):
    insights = []
    
    # Savings insights
    savings_ratio = savings_amount / input_data['Income'] if input_data['Income'] > 0 else 0
    
    if savings_ratio > 0.15:
        insights.append("Strong savings behavior: This household saves more than 15% of income.")
    elif savings_ratio > 0.1:
        insights.append("Moderate savings: This household maintains a savings rate of 10-15%.")
    elif savings_ratio > 0.05:
        insights.append("Basic savings: This household saves 5-10% of income, but could improve.")
    else:
        insights.append("Low savings: This household saves less than 5% of income, indicating vulnerability.")
    
    # Debt insights
    if input_data['Debt_to_Income'] > 0.6:
        insights.append("High debt burden: Debt exceeds 60% of income, limiting financial flexibility.")
    elif input_data['Debt_to_Income'] > 0.3:
        insights.append("Moderate debt levels: Manageable but should be monitored.")
    else:
        insights.append("Low debt levels: Healthy financial position with minimal debt burden.")
    
    # Emergency fund insights
    if input_data['Emergency_Ratio'] > 0.25:
        insights.append("Strong emergency fund: Household has significant reserves for unexpected expenses.")
    elif input_data['Emergency_Ratio'] > 0.1:
        insights.append("Adequate emergency fund: Some protection against income shocks.")
    else:
        insights.append("Limited emergency fund: Vulnerable to unexpected financial emergencies.")
    
    # Employment insights
    if input_data['Employment_Status'] == 'Employed':
        insights.append("Stable employment: Regular income source supports consistent savings.")
    elif input_data['Employment_Status'] == 'Self-Employed':
        insights.append("Variable income: Self-employment may lead to irregular savings patterns.")
    else:
        insights.append("Income uncertainty: Lack of stable employment may hinder savings capacity.")
    
    # Regional insights
    if input_data['Poverty_Rate'] > 0.25:
        insights.append("High-poverty region: External economic factors may constrain savings opportunities.")
    elif input_data['Poverty_Rate'] > 0.15:
        insights.append("Moderate-poverty region: Some community-level economic challenges exist.")
    else:
        insights.append("Lower-poverty region: Favorable economic environment supports savings behavior.")
    
    # Resilience insights
    if resilience_score > 75:
        insights.append("High financial resilience: This household is well-prepared to withstand economic shocks.")
    elif resilience_score > 50:
        insights.append("Moderate financial resilience: Some capacity to handle financial challenges.")
    else:
        insights.append("Low financial resilience: Vulnerable to income shocks and emergencies.")
    
    return insights

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">Savings Behavior & Financial Resilience Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        Integrating public data with household-level insights to model savings behavior of daily wage earners
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    artifacts = load_model()
    
    # Create tabs - Updated to include scheme recommendations
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Household Analysis", 
        "Scheme Recommendations",
        "Regional Insights", 
        "Policy Impact", 
        "Government Schemes (English)", 
        "सरकारी योजनाएं (Hindi)",
        "शासकीय योजना (Marathi)"
    ])
    
    with tab1:
        st.markdown('<div class="sub-header">Household Savings Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Financial Information")
            income = st.number_input("Monthly Income (₹)", min_value=0, step=1000)
            st.markdown('<div class="max-limit">Max limit: No limit</div>', unsafe_allow_html=True)
            
            debt_to_income = st.number_input("Debt to Income Ratio", min_value=0.0, max_value=1.0, step=0.01,
                                          help="Total monthly debt payments divided by monthly income")
            st.markdown('<div class="max-limit">Max limit: 1.00</div>', unsafe_allow_html=True)
            
            expense_ratio = st.number_input("Expense Ratio", min_value=0.0, max_value=1.0, step=0.01,
                                         help="Monthly expenses divided by monthly income")
            st.markdown('<div class="max-limit">Max limit: 1.00</div>', unsafe_allow_html=True)
            
            emergency_ratio = st.number_input("Emergency Fund Ratio", min_value=0.0, max_value=1.0, step=0.01,
                                           help="Emergency savings divided by monthly income")
            st.markdown('<div class="max-limit">Max limit: 1.00</div>', unsafe_allow_html=True)
            
            avg_monthly_consumption = st.number_input("Average Monthly Consumption (₹)", min_value=0, step=1000)
            st.markdown('<div class="max-limit">Max limit: No limit</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("Demographic & Regional Information")
            education = st.selectbox("Education Level", ["High School", "Graduate", "Post Graduate", "Doctorate"])
            employment_status = st.selectbox("Employment Status", ["Unemployed", "Employed", "Self-Employed", "Retired"])
            designation = st.selectbox("Occupation Type", ["Daily Wage", "Skilled Labor", "Clerical", "Supervisory", "Management"])
            
            cpi_index = st.number_input("Regional CPI Index", min_value=90, max_value=130, step=1,
                                     help="Consumer Price Index for the region (100 = national average)")
            st.markdown('<div class="max-limit">Max limit: 130</div>', unsafe_allow_html=True)
            
            avg_mgnrega_days = st.number_input("Average MGNREGA Days (yearly)", min_value=0, max_value=100, step=1,
                                            help="Average days of employment under MGNREGA scheme")
            st.markdown('<div class="max-limit">Max limit: 100</div>', unsafe_allow_html=True)
            
            poverty_rate = st.number_input("Regional Poverty Rate", min_value=0.0, max_value=0.5, step=0.01,
                                        help="Percentage of population below poverty line in the region")
            st.markdown('<div class="max-limit">Max limit: 0.50</div>', unsafe_allow_html=True)
        
        # Additional inputs for better recommendations
        st.subheader("Additional Information")
        col3, col4 = st.columns(2)
        with col3:
            has_bank_account = st.checkbox("Has Bank Account", value=True)
            has_girl_child = st.checkbox("Has Girl Child (below 10 years)")
        with col4:
            age_group = st.selectbox("Age Group", ["18-25", "26-35", "36-45", "46-55", "Above 55"])
            has_life_insurance = st.checkbox("Has Life Insurance")
        
        # Prepare input data
        input_data = {
            "Income": income,
            "Debt_to_Income": debt_to_income,
            "Expense_Ratio": expense_ratio,
            "Emergency_Ratio": emergency_ratio,
            "Education": education,
            "Employment_Status": employment_status,
            "Designation": designation,
            "CPI_Index": cpi_index,
            "Avg_Monthly_Consumption": avg_monthly_consumption,
            "Avg_MGNREGA_Days": avg_mgnrega_days,
            "Poverty_Rate": poverty_rate,
            "Has_Bank_Account": has_bank_account,
            "Has_Girl_Child": has_girl_child,
            "Age_Group": age_group,
            "Has_Life_Insurance": has_life_insurance
        }
        
        # Store data in session state for use in other tabs
        st.session_state['input_data'] = input_data
        
        # Calculate results
        if st.button("Analyze Savings Behavior", type="primary"):
            if income == 0:
                st.error("Monthly income cannot be zero. Please enter a valid income amount.")
                return
            
            # Estimate savings amount
            savings_amount, savings_rate = estimate_savings_amount(input_data)
            
            # Calculate resilience score
            resilience_score = calculate_resilience_score(input_data, savings_amount)
            
            # Generate insights
            insights = generate_insights(input_data, savings_amount, resilience_score)
            
            # Store results in session state
            st.session_state['savings_amount'] = savings_amount
            st.session_state['savings_rate'] = savings_rate
            st.session_state['resilience_score'] = resilience_score
            st.session_state['insights'] = insights
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Estimated Monthly Savings", f"₹{savings_amount:,.0f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Savings Rate", f"{savings_rate*100:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Financial Resilience Score", f"{resilience_score:.0f}/100")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Savings distribution chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Income', 'Expenses', 'Savings'],
                y=[income, income * expense_ratio, savings_amount],
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
            ))
            fig.update_layout(
                title="Income Distribution",
                xaxis_title="Category",
                yaxis_title="Amount (₹)",
                showlegend=False
            )
            st.plotly_chart(fig, width='stretch')
            
            # Display insights
            st.markdown('<div class="sub-header">Key Insights</div>', unsafe_allow_html=True)
            for insight in insights:
                st.markdown(f'<div class="insight-box">• {insight}</div>', unsafe_allow_html=True)
            
            # Quick scheme recommendations teaser
            st.markdown('<div class="sub-header">Quick Scheme Recommendations</div>', unsafe_allow_html=True)
            user_profile = determine_user_profile(input_data, savings_amount, resilience_score)
            quick_recommendations = recommend_schemes(user_profile, input_data, savings_amount)[:3]
            
            for rec in quick_recommendations:
                priority_class = f"priority-{rec['priority'].lower()}"
                st.markdown(f'''
                <div class="scheme-card {priority_class}">
                    <h4>{rec['scheme']['name']} ({rec['scheme']['type']})</h4>
                    <p><strong>Priority:</strong> {rec['priority']} | <strong>Match Score:</strong> {rec['score']}/100</p>
                    <p><strong>Key Benefits:</strong> {', '.join(rec['scheme']['benefits'][:2])}</p>
                    <p><strong>Why recommended:</strong> {rec['reasons'][0] if rec['reasons'] else 'Good fit for your profile'}</p>
                </div>
                ''', unsafe_allow_html=True)
            
            st.info(" Visit the 'Scheme Recommendations' tab for detailed analysis and personalized suggestions!")
            
            # Resilience breakdown
            st.markdown('<div class="sub-header">Resilience Factors</div>', unsafe_allow_html=True)
            factors = ['Income Stability', 'Debt Management', 'Emergency Fund', 'Savings Rate', 'Regional Economy']
            scores = [
                min(100, max(0, (income / 30000) * 25)),  # Income factor
                max(0, 25 - (debt_to_income * 35)),       # Debt factor
                min(25, emergency_ratio * 100),           # Emergency fund factor
                min(25, savings_rate * 150),              # Savings factor
                max(0, 25 - (poverty_rate * 100))         # Regional factor
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=scores,
                y=factors,
                orientation='h',
                marker_color='#1f77b4'
            ))
            fig.update_layout(
                title="Financial Resilience Components",
                xaxis_title="Score (0-25)",
                yaxis_title="Factor",
                xaxis_range=[0, 25],
                height=300
            )
            st.plotly_chart(fig, width='stretch')
    
    with tab2:
        st.markdown('<div class="sub-header"> Personalized Government Scheme Recommendations</div>', unsafe_allow_html=True)
        
        # Check if analysis has been run
        if 'input_data' not in st.session_state:
            st.warning("Please run the analysis in the 'Household Analysis' tab first to get personalized recommendations.")
            return
        
        # Get data from session state
        input_data = st.session_state.get('input_data', {})
        savings_amount = st.session_state.get('savings_amount', 0)
        resilience_score = st.session_state.get('resilience_score', 0)
        
        if savings_amount == 0:
            st.warning("Please run the analysis in the 'Household Analysis' tab first to get personalized recommendations.")
            return
        
        # Generate recommendations
        user_profile = determine_user_profile(input_data, savings_amount, resilience_score)
        recommendations = recommend_schemes(user_profile, input_data, savings_amount)
        
        # Display user profile summary
        st.markdown('<div class="sub-header">Your Financial Profile</div>', unsafe_allow_html=True)
        
        profile_cols = st.columns(4)
        with profile_cols[0]:
            st.metric("Income Level", user_profile['income_level'].title())
            st.metric("Employment Type", user_profile['employment_type'].title())
        with profile_cols[1]:
            st.metric("Savings Behavior", user_profile['savings_behavior'].title())
            st.metric("Financial Stability", user_profile['financial_stability'].title())
        with profile_cols[2]:
            st.metric("Debt Burden", user_profile['debt_burden'].title())
            st.metric("Emergency Preparedness", user_profile['emergency_preparedness'].title())
        with profile_cols[3]:
            st.metric("Monthly Income", f"₹{input_data['Income']:,}")
            st.metric("Monthly Savings", f"₹{savings_amount:,.0f}")
        
        # Filter and display recommendations
        st.markdown('<div class="sub-header"> Recommended Schemes for You</div>', unsafe_allow_html=True)
        
        if not recommendations:
            st.info("No specific scheme recommendations available. Please review the general schemes in other tabs.")
            return
        
        # Priority filter
        priority_filter = st.selectbox("Filter by Priority", ["All", "High", "Medium", "Low"])
        
        filtered_recommendations = recommendations
        if priority_filter != "All":
            filtered_recommendations = [r for r in recommendations if r['priority'] == priority_filter]
        
        # Display recommendations
        for i, rec in enumerate(filtered_recommendations):
            priority_class = f"priority-{rec['priority'].lower()}"
            
            # Create expandable sections for each recommendation
            with st.expander(f" {rec['scheme']['name']} - {rec['priority']} Priority (Score: {rec['score']}/100)", expanded=(i < 2)):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Type:** {rec['scheme']['type']}")
                    st.markdown(f"**Website:** [{rec['scheme']['website']}]({rec['scheme']['website']})")
                    
                    # Benefits
                    st.markdown("**Key Benefits:**")
                    for benefit in rec['scheme']['benefits']:
                        st.markdown(f"• {benefit}")
                    
                    # Why recommended
                    st.markdown("**Why this scheme is recommended for you:**")
                    for reason in rec['reasons']:
                        st.markdown(f"✓ {reason}")
                
                with col2:
                    # Priority indicator
                    priority_color = "#ffc107" if rec['priority'] == "High" else "#17a2b8" if rec['priority'] == "Medium" else "#6c757d"
                    st.markdown(f'''
                    <div class="metric-box" style="border-left: 5px solid {priority_color};">
                        <h3 class="recommendation-score">{rec['priority']} Priority</h3>
                        <p>Match Score: {rec['score']}/100</p>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Estimated benefits
                    if rec['estimated_benefit']:
                        st.markdown("**Estimated Benefits:**")
                        benefits = rec['estimated_benefit']
                        
                        if 'annual_cost' in benefits:
                            st.metric("Annual Cost", f"₹{benefits['annual_cost']:,.0f}")
                        if 'insurance_cover' in benefits:
                            st.metric("Insurance Cover", f"₹{benefits['insurance_cover']:,.0f}")
                        if 'overdraft_limit' in benefits:
                            st.metric("Overdraft Limit", f"₹{benefits['overdraft_limit']:,.0f}")
                        if 'monthly_pension_at_60' in benefits:
                            st.metric("Monthly Pension", f"₹{benefits['monthly_pension_at_60']:,.0f}")
                        if 'estimated_maturity_value' in benefits:
                            st.metric("Maturity Value", f"₹{benefits['estimated_maturity_value']:,.0f}")
                        if 'annual_tax_savings' in benefits:
                            st.metric("Annual Tax Savings", f"₹{benefits['annual_tax_savings']:,.0f}")
        
        # Recommendation summary chart
        if recommendations:
            st.markdown('<div class="sub-header">Recommendation Summary</div>', unsafe_allow_html=True)
            
            # Create summary data
            scheme_names = [rec['scheme']['name'][:20] + '...' if len(rec['scheme']['name']) > 20 
                          else rec['scheme']['name'] for rec in recommendations]
            scores = [rec['score'] for rec in recommendations]
            priorities = [rec['priority'] for rec in recommendations]
            
            # Color mapping for priorities
            color_map = {'High': '#ffc107', 'Medium': '#17a2b8', 'Low': '#6c757d'}
            colors = [color_map[priority] for priority in priorities]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=scores,
                y=scheme_names,
                orientation='h',
                marker_color=colors,
                text=[f"{score}" for score in scores],
                textposition='inside'
            ))
            
            fig.update_layout(
                title="Scheme Recommendation Scores",
                xaxis_title="Recommendation Score (0-100)",
                yaxis_title="Government Schemes",
                height=max(400, len(recommendations) * 50),
                showlegend=False
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # Action plan
            st.markdown('<div class="sub-header"> Your Action Plan</div>', unsafe_allow_html=True)
            
            high_priority = [r for r in recommendations if r['priority'] == 'High']
            medium_priority = [r for r in recommendations if r['priority'] == 'Medium']
            
            if high_priority:
                st.markdown("**Immediate Actions (High Priority):**")
                for i, rec in enumerate(high_priority[:3], 1):
                    st.markdown(f"{i}. **{rec['scheme']['name']}** - {rec['reasons'][0] if rec['reasons'] else 'Highly suitable for your profile'}")
                    st.markdown(f"    Visit: [{rec['scheme']['website']}]({rec['scheme']['website']})")
            
            if medium_priority:
                st.markdown("**Consider Later (Medium Priority):**")
                for i, rec in enumerate(medium_priority[:2], 1):
                    st.markdown(f"{i}. **{rec['scheme']['name']}** - {rec['reasons'][0] if rec['reasons'] else 'Good option to consider'}")
            
            # Financial impact summary
            st.markdown('<div class="sub-header"> Potential Financial Impact</div>', unsafe_allow_html=True)
            
            total_insurance_cover = sum([rec['estimated_benefit'].get('insurance_cover', 0) + 
                                       rec['estimated_benefit'].get('life_cover', 0) + 
                                       rec['estimated_benefit'].get('accidental_cover', 0) 
                                       for rec in recommendations if rec['estimated_benefit']])
            
            total_annual_cost = sum([rec['estimated_benefit'].get('annual_cost', 0) + 
                                   rec['estimated_benefit'].get('annual_premium', 0) 
                                   for rec in recommendations if rec['estimated_benefit']])
            
            total_tax_savings = sum([rec['estimated_benefit'].get('annual_tax_savings', 0) 
                                   for rec in recommendations if rec['estimated_benefit']])
            
            impact_cols = st.columns(3)
            with impact_cols[0]:
                st.metric("Total Insurance Coverage", f"₹{total_insurance_cover:,.0f}")
            with impact_cols[1]:
                st.metric("Total Annual Investment", f"₹{total_annual_cost:,.0f}")
            with impact_cols[2]:
                st.metric("Potential Tax Savings", f"₹{total_tax_savings:,.0f}")
            
            # Affordability check
            if total_annual_cost > 0:
                affordability_ratio = total_annual_cost / (input_data['Income'] * 12)
                if affordability_ratio > 0.1:
                    st.warning(f" The recommended schemes require {affordability_ratio*100:.1f}% of your annual income. Consider starting with high-priority schemes only.")
                elif affordability_ratio > 0.05:
                    st.info(f" The recommended schemes require {affordability_ratio*100:.1f}% of your annual income, which is reasonable for financial security.")
                else:
                    st.success(f" The recommended schemes require only {affordability_ratio*100:.1f}% of your annual income, making them very affordable.")
    
    with tab3:
        st.markdown('<div class="sub-header">Regional Economic Insights</div>', unsafe_allow_html=True)
        
        # Simulated regional data
        regions = ['North', 'South', 'East', 'West', 'Central']
        poverty_rates = [0.28, 0.18, 0.32, 0.22, 0.35]
        avg_incomes = [22000, 28000, 19000, 25000, 18000]
        avg_savings_rates = [0.08, 0.14, 0.06, 0.11, 0.05]
        mgnrega_days = [55, 35, 60, 40, 65]
        
        region_data = pd.DataFrame({
            'Region': regions,
            'Poverty_Rate': poverty_rates,
            'Avg_Income': avg_incomes,
            'Avg_Savings_Rate': avg_savings_rates,
            'MGNREGA_Days': mgnrega_days
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(region_data, x='Region', y='Avg_Income', 
                         title='Average Monthly Income by Region',
                         color='Region')
            st.plotly_chart(fig, width='stretch')
            
            fig = px.scatter(region_data, x='Poverty_Rate', y='Avg_Savings_Rate',
                             size='MGNREGA_Days', color='Region',
                             title='Savings Rate vs Poverty Rate',
                             labels={'Poverty_Rate': 'Poverty Rate', 'Avg_Savings_Rate': 'Average Savings Rate'})
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            fig = px.bar(region_data, x='Region', y='Avg_Savings_Rate', 
                         title='Average Savings Rate by Region',
                         color='Region')
            st.plotly_chart(fig, width='stretch')
            
            fig = px.line(region_data, x='Region', y='MGNREGA_Days',
                          title='MGNREGA Employment Days by Region',
                          markers=True)
            st.plotly_chart(fig, width='stretch')
        
        st.markdown('<div class="sub-header">Regional Correlation Analysis</div>', unsafe_allow_html=True)
        st.write("""
        The charts above show how regional economic factors influence savings behavior:
        - Regions with lower poverty rates tend to have higher savings rates
        - MGNREGA employment provides income stability that supports savings
        - Higher average incomes are correlated with higher savings rates
        """)
    
    with tab4:
        st.markdown('<div class="sub-header">Policy Impact Simulation</div>', unsafe_allow_html=True)
        
        st.write("Simulate how different policy interventions might affect savings behavior:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Scenario")
            current_savings = st.session_state.get('savings_amount', 3750)
            current_rate = st.session_state.get('savings_rate', 0.15)
            current_resilience = st.session_state.get('resilience_score', 68)
            
            st.metric("Estimated Monthly Savings", f"₹{current_savings:,.0f}")
            st.metric("Savings Rate", f"{current_rate*100:.1f}%")
            st.metric("Financial Resilience Score", f"{current_resilience:.0f}/100")
        
        with col2:
            st.subheader("Policy Options")
            policy_option = st.selectbox(
                "Select policy intervention",
                ["None", "MGNREGA Expansion", "Financial Literacy", "Debt Relief", "Savings Incentive"]
            )
            
            if policy_option == "MGNREGA Expansion":
                new_savings = current_savings * 1.1
                new_rate = current_rate * 1.1
                new_resilience = current_resilience + 4
                st.metric("Projected Monthly Savings", f"₹{new_savings:,.0f}", f"+{((new_savings/current_savings-1)*100):.0f}%")
                st.metric("Projected Savings Rate", f"{new_rate*100:.1f}%", f"+{((new_rate-current_rate)*100):.1f}%")
                st.metric("Projected Resilience Score", f"{new_resilience:.0f}/100", f"+{new_resilience-current_resilience}")
            elif policy_option == "Financial Literacy":
                new_savings = current_savings * 1.2
                new_rate = current_rate * 1.2
                new_resilience = current_resilience + 7
                st.metric("Projected Monthly Savings", f"₹{new_savings:,.0f}", f"+{((new_savings/current_savings-1)*100):.0f}%")
                st.metric("Projected Savings Rate", f"{new_rate*100:.1f}%", f"+{((new_rate-current_rate)*100):.1f}%")
                st.metric("Projected Resilience Score", f"{new_resilience:.0f}/100", f"+{new_resilience-current_resilience}")
            elif policy_option == "Debt Relief":
                new_savings = current_savings * 1.3
                new_rate = current_rate * 1.3
                new_resilience = current_resilience + 10
                st.metric("Projected Monthly Savings", f"₹{new_savings:,.0f}", f"+{((new_savings/current_savings-1)*100):.0f}%")
                st.metric("Projected Savings Rate", f"{new_rate*100:.1f}%", f"+{((new_rate-current_rate)*100):.1f}%")
                st.metric("Projected Resilience Score", f"{new_resilience:.0f}/100", f"+{new_resilience-current_resilience}")
            elif policy_option == "Savings Incentive":
                new_savings = current_savings * 1.4
                new_rate = current_rate * 1.4
                new_resilience = current_resilience + 12
                st.metric("Projected Monthly Savings", f"₹{new_savings:,.0f}", f"+{((new_savings/current_savings-1)*100):.0f}%")
                st.metric("Projected Savings Rate", f"{new_rate*100:.1f}%", f"+{((new_rate-current_rate)*100):.1f}%")
                st.metric("Projected Resilience Score", f"{new_resilience:.0f}/100", f"+{new_resilience-current_resilience}")
            else:
                st.metric("Projected Monthly Savings", f"₹{current_savings:,.0f}", "0%")
                st.metric("Projected Savings Rate", f"{current_rate*100:.1f}%", "0.0%")
                st.metric("Projected Resilience Score", f"{current_resilience:.0f}/100", "0")
        
        st.markdown('<div class="sub-header">Policy Impact Explanation</div>', unsafe_allow_html=True)
        if policy_option == "MGNREGA Expansion":
            st.write("""
            **MGNREGA Expansion**: Increasing guaranteed employment days provides more stable income,
            reducing vulnerability to seasonal fluctuations and supporting consistent savings behavior.
            """)
        elif policy_option == "Financial Literacy":
            st.write("""
            **Financial Literacy Programs**: Education on budgeting, saving, and debt management
            helps households make better financial decisions and increase savings rates.
            """)
        elif policy_option == "Debt Relief":
            st.write("""
            **Debt Relief Programs**: Reducing debt burdens frees up income for savings and
            reduces financial stress, improving overall financial resilience.
            """)
        elif policy_option == "Savings Incentive":
            st.write("""
            **Savings Incentives**: Matching programs or interest subsidies encourage higher
            savings rates and help households build financial buffers more quickly.
            """)
        else:
            st.write("Select a policy option to see its projected impact on savings behavior.")

    with tab5:
        st.markdown('<div class="sub-header">Government Savings & Social Security Schemes</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box">
            All these schemes fall under the <strong>Jan Suraksha (Social Security)</strong> umbrella.
            More details available at: <a href="https://jansuraksha.gov.in" target="_blank">https://jansuraksha.gov.in</a>
        </div>
        """, unsafe_allow_html=True)
        
        # PMJDY
        st.markdown("""
        <div class="scheme-card">
            <h3>1. Pradhan Mantri Jan Dhan Yojana (PMJDY)</h3>
            <p><strong>Website:</strong> <a href="https://pmjdy.gov.in" target="_blank">https://pmjdy.gov.in</a></p>
            <p><strong>Details:</strong></p>
            <ul>
                <li>National Mission for Financial Inclusion</li>
                <li>Provides zero-balance savings account, free RuPay debit card, overdraft facility, and insurance cover</li>
                <li>Target group: Unbanked poor, daily wage earners, rural and urban poor</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # APY
        st.markdown("""
        <div class="scheme-card">
            <h3>2. Atal Pension Yojana (APY)</h3>
            <p><strong>Website:</strong> <a href="https://npstrust.org.in/content/atal-pension-yojana" target="_blank">https://npstrust.org.in/content/atal-pension-yojana</a></p>
            <p><strong>Details:</strong></p>
            <ul>
                <li>Pension scheme for unorganized sector workers</li>
                <li>Subscribers (18–40 years) contribute monthly, and receive a fixed pension (₹1000–₹5000 per month) after 60 years of age</li>
                <li>Government co-contribution available for eligible low-income subscribers</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # PPF
        st.markdown("""
        <div class="scheme-card">
            <h3>3. Public Provident Fund (PPF)</h3>
            <p><strong>Website:</strong> <a href="https://www.indiapost.gov.in" target="_blank">https://www.indiapost.gov.in</a></p>
            <p><strong>Details:</strong></p>
            <ul>
                <li>Long-term savings-cum-tax saving scheme</li>
                <li>Tenure: 15 years (extendable in blocks of 5 years)</li>
                <li>Annual deposit: ₹500 – ₹1.5 lakh</li>
                <li>Interest rate: ~7.1% (compounded annually, tax-free)</li>
                <li>Tax benefits: EEE (investment, interest, and maturity all exempt under 80C)</li>
                <li>Available to all Indian residents</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # SSY
        st.markdown("""
        <div class="scheme-card">
            <h3>4. Sukanya Samriddhi Yojana (SSY)</h3>
            <p><strong>Website:</strong> <a href="https://www.indiapost.gov.in" target="_blank">https://www.indiapost.gov.in</a></p>
            <p><strong>Details:</strong></p>
            <ul>
                <li>Savings scheme for girl child</li>
                <li>Eligible: Girl below 10 years + parents/guardians as account holders</li>
                <li>Tenure: Till girl turns 21 or marriage after 18</li>
                <li>Annual deposit: ₹250 – ₹1.5 lakh</li>
                <li>Interest rate: ~8.2% (compounded annually, tax-free)</li>
                <li>Tax benefits: 80C deduction</li>
                <li>Only one account per girl child (max 2 per family, exceptions for twins/triplets)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional schemes
        st.markdown("""
        <div class="scheme-card">
            <h3>5. Pradhan Mantri Suraksha Bima Yojana (PMSBY)</h3>
            <p><strong>Website:</strong> <a href="https://jansuraksha.gov.in" target="_blank">https://jansuraksha.gov.in</a></p>
            <p><strong>Details:</strong></p>
            <ul>
                <li>Accidental death and disability insurance scheme</li>
                <li>Coverage: ₹2 lakh for accidental death/permanent total disability</li>
                <li>Premium: ₹20 per year (auto-debit from bank account)</li>
                <li>Age group: 18-70 years</li>
                <li>Renewable annually</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="scheme-card">
            <h3>6. Pradhan Mantri Jeevan Jyoti Bima Yojana (PMJJBY)</h3>
            <p><strong>Website:</strong> <a href="https://jansuraksha.gov.in" target="_blank">https://jansuraksha.gov.in</a></p>
            <p><strong>Details:</strong></p>
            <ul>
                <li>Life insurance scheme</li>
                <li>Coverage: ₹2 lakh life insurance cover</li>
                <li>Premium: ₹330 per year (auto-debit from bank account)</li>
                <li>Age group: 18-50 years (renewable up to 55 years)</li>
                <li>Risk coverage for any reason</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Scheme comparison
        st.markdown('<div class="sub-header">Scheme Comparison</div>', unsafe_allow_html=True)
        
        scheme_data = pd.DataFrame({
            'Scheme': ['PMJDY', 'APY', 'PPF', 'SSY', 'PMSBY', 'PMJJBY'],
            'Type': ['Banking', 'Pension', 'Savings', 'Child Savings', 'Accident Insurance', 'Life Insurance'],
            'Min Investment (₹)': ['Zero', 'Variable', '500/year', '250/year', '20/year', '330/year'],
            'Max Investment (₹)': ['No limit', 'No limit', '1.5 lakh/year', '1.5 lakh/year', 'Fixed', 'Fixed'],
            'Key Benefit': ['Bank account + insurance', '₹1000-5000/month pension', '7.1% tax-free returns', '8.2% tax-free returns', '₹2 lakh accident cover', '₹2 lakh life cover'],
            'Target Group': ['Unbanked population', 'Unorganized workers', 'All residents', 'Girl children <10 years', 'Age 18-70', 'Age 18-50']
        })
        
        st.dataframe(scheme_data, width='stretch')
        
        # Eligibility guidance
        st.markdown('<div class="sub-header">Which Scheme is Right For You?</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="insight-box">
                <h4>For Basic Banking Needs</h4>
                <p><strong>PMJDY</strong> is ideal if you don't have a bank account yet. 
                It provides a zero-balance account with debit card and insurance coverage.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="insight-box">
                <h4>For Long-term Tax-free Savings</h4>
                <p><strong>Public Provident Fund (PPF)</strong> is perfect for long-term savings 
                with tax benefits and guaranteed ~7.1% returns.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="insight-box">
                <h4>For Insurance Protection</h4>
                <p><strong>PMSBY & PMJJBY</strong> provide essential insurance coverage at very low cost. 
                Ideal for daily wage earners and their families.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="insight-box">
                <h4>For Retirement Planning</h4>
                <p><strong>Atal Pension Yojana</strong> is perfect if you're 18-40 years old 
                and want to secure pension after 60 years of age.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="insight-box">
                <h4>For Girl Child's Education</h4>
                <p><strong>Sukanya Samriddhi Yojana (SSY)</strong> offers the highest returns (~8.2%) 
                for saving towards your daughter's education and marriage.</p>
            </div>
            """, unsafe_allow_html=True)

    with tab6:
        st.markdown('<div class="sub-header">सरकारी बचत और सामाजिक सुरक्षा योजनाएं</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box">
            ये सभी योजनाएं <strong>जन सुरक्षा (सामाजिक सुरक्षा)</strong> के अंतर्गत आती हैं।
            अधिक जानकारी के लिए: <a href="https://jansuraksha.gov.in" target="_blank">https://jansuraksha.gov.in</a>
        </div>
        """, unsafe_allow_html=True)
        
        # PMJDY in Hindi
        st.markdown("""
        <div class="scheme-card">
            <h3>1. प्रधानमंत्री जन धन योजना (PMJDY)</h3>
            <p><strong>वेबसाइट:</strong> <a href="https://pmjdy.gov.in" target="_blank">https://pmjdy.gov.in</a></p>
            <p><strong>विवरण:</strong></p>
            <ul>
                <li>वित्तीय समावेशन के लिए राष्ट्रीय मिशन</li>
                <li>जीरो-बैलेंस बचत खाता, मुफ्त रुपे डेबिट कार्ड, ओवरड्राफ्ट सुविधा और बीमा कवर प्रदान करता है</li>
                <li>लक्ष्य समूह: अबैंक गरीब, दैनिक मजदूरी कमाने वाले, ग्रामीण और शहरी गरीब</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # APY in Hindi
        st.markdown("""
        <div class="scheme-card">
            <h3>2. अटल पेंशन योजना (APY)</h3>
            <p><strong>वेबसाइट:</strong> <a href="https://npstrust.org.in/content/atal-pension-yojana" target="_blank">https://npstrust.org.in/content/atal-pension-yojana</a></p>
            <p><strong>विवरण:</strong></p>
            <ul>
                <li>असंगठित क्षेत्र के श्रमिकों के लिए पेंशन योजना</li>
                <li>ग्राहक (18-40 वर्ष) मासिक योगदान देते हैं, और 60 वर्ष की आयु के बाद निश्चित पेंशन (₹1000-₹5000 प्रति माह) प्राप्त करते हैं</li>
                <li>योग्य कम आय वाले ग्राहकों के लिए सरकारी सह-योगदान उपलब्ध</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # PPF in Hindi
        st.markdown("""
        <div class="scheme-card">
            <h3>3. पब्लिक प्रोविडेंट फंड (PPF)</h3>
            <p><strong>वेबसाइट:</strong> <a href="https://www.indiapost.gov.in" target="_blank">https://www.indiapost.gov.in</a></p>
            <p><strong>विवरण:</strong></p>
            <ul>
                <li>दीर्घकालिक बचत-कम-टैक्स बचत योजना</li>
                <li>अवधि: 15 वर्ष (5 वर्ष के ब्लॉक में विस्तार योग्य)</li>
                <li>वार्षिक जमा: ₹500 - ₹1.5 लाख</li>
                <li>ब्याज दर: ~7.1% (वार्षिक चक्रवृद्धि, कर-मुक्त)</li>
                <li>टैक्स लाभ: ईईई (निवेश, ब्याज और परिपक्वता सभी 80C के तहत छूट)</li>
                <li>सभी भारतीय निवासियों के लिए उपलब्ध</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # SSY in Hindi
        st.markdown("""
        <div class="scheme-card">
            <h3>4. सुकन्या समृद्धि योजना (SSY)</h3>
            <p><strong>वेबसाइट:</strong> <a href="https://www.indiapost.gov.in" target="_blank">https://www.indiapost.gov.in</a></p>
            <p><strong>विवरण:</strong></p>
            <ul>
                <li>बालिका के लिए बचत योजना</li>
                <li>योग्य: 10 वर्ष से कम उम्र की बालिका + माता-पिता/अभिभावक खाताधारक के रूप में</li>
                <li>अवधि: बालिका के 21 वर्ष की आयु तक या 18 वर्ष के बाद विवाह</li>
                <li>वार्षिक जमा: ₹250 - ₹1.5 लाख</li>
                <li>ब्याज दर: ~8.2% (वार्षिक चक्रवृद्धि, कर-मुक्त)</li>
                <li>टैक्स लाभ: 80C कटौती</li>
                <li>प्रति बालिका केवल एक खाता (अधिकतम 2 प्रति परिवार, जुड़वाँ/त्रिक के लिए अपवाद)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional schemes in Hindi
        st.markdown("""
        <div class="scheme-card">
            <h3>5. प्रधानमंत्री सुरक्षा बीमा योजना (PMSBY)</h3>
            <p><strong>वेबसाइट:</strong> <a href="https://jansuraksha.gov.in" target="_blank">https://jansuraksha.gov.in</a></p>
            <p><strong>विवरण:</strong></p>
            <ul>
                <li>आकस्मिक मृत्यु और विकलांगता बीमा योजना</li>
                <li>कवरेज: आकस्मिक मृत्यु/स्थायी कुल विकलांगता के लिए ₹2 लाख</li>
                <li>प्रीमियम: ₹20 प्रति वर्ष (बैंक खाते से ऑटो-डेबिट)</li>
                <li>आयु वर्ग: 18-70 वर्ष</li>
                <li>वार्षिक नवीनीकरण योग्य</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="scheme-card">
            <h3>6. प्रधानमंत्री जीवन ज्योति बीमा योजना (PMJJBY)</h3>
            <p><strong>वेबसाइट:</strong> <a href="https://jansuraksha.gov.in" target="_blank">https://jansuraksha.gov.in</a></p>
            <p><strong>विवरण:</strong></p>
            <ul>
                <li>जीवन बीमा योजना</li>
                <li>कवरेज: ₹2 लाख जीवन बीमा कवर</li>
                <li>प्रीमियम: ₹330 प्रति वर्ष (बैंक खाते से ऑटो-डेबिट)</li>
                <li>आयु वर्ग: 18-50 वर्ष (55 वर्ष तक नवीनीकरण योग्य)</li>
                <li>किसी भी कारण से जोखिम कवरेज</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab7:
        st.markdown('<div class="sub-header">शासकीय बचत आणि सामाजिक सुरक्षा योजना</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box">
            ह्या सर्व योजना <strong>जन सुरक्षा (सामाजिक सुरक्षा)</strong> अंतर्गत येतात.
            अधिक माहितीसाठी: <a href="https://jansuraksha.gov.in" target="_blank">https://jansuraksha.gov.in</a>
        </div>
        """, unsafe_allow_html=True)
        
        # PMJDY in Marathi
        st.markdown("""
        <div class="scheme-card">
            <h3>1. प्रधानमंत्री जन धन योजना (PMJDY)</h3>
            <p><strong>वेबसाइट:</strong> <a href="https://pmjdy.gov.in" target="_blank">https://pmjdy.gov.in</a></p>
            <p><strong>तपशील:</strong></p>
            <ul>
                <li>आर्थिक समावेशनासाठी राष्ट्रीय मिशन</li>
                <li>शून्य-शिल्लक बचत खाते, विनामूल्य रुपे डेबिट कार्ड, ओव्हरड्राफ्ट सुविधा आणि विमा कव्हर प्रदान करते</li>
                <li>लक्ष्य गट: बॅंकेत नसलेले गरीब, दैनिक मजुरी कमावणारे, ग्रामीण आणि शहरी गरीब</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # APY in Marathi
        st.markdown("""
        <div class="scheme-card">
            <h3>2. अटल पेन्शन योजना (APY)</h3>
            <p><strong>वेबसाइट:</strong> <a href="https://npstrust.org.in/content/atal-pension-yojana" target="_blank">https://npstrust.org.in/content/atal-pension-yojana</a></p>
            <p><strong>तपशील:</strong></p>
            <ul>
                <li>असंघटित क्षेत्रातील कामगारांसाठी पेन्शन योजना</li>
                <li>सदस्य (18-40 वर्षे) मासिक योगदान देतात, आणि 60 वर्षांच्या वयानंतर निश्चित पेन्शन (₹1000-₹5000 प्रति महिना) मिळवतात</li>
                <li>पात्र कमी उत्पन्न असलेल्या सदस्यांसाठी सरकारी सह-योगदान उपलब्ध</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # PPF in Marathi
        st.markdown("""
        <div class="scheme-card">
            <h3>3. पब्लिक प्रोव्हिडेंट फंड (PPF)</h3>
            <p><strong>वेबसाइट:</strong> <a href="https://www.indiapost.gov.in" target="_blank">https://www.indiapost.gov.in</a></p>
            <p><strong>तपशील:</strong></p>
            <ul>
                <li>दीर्घकालीन बचत-कर-बचत योजना</li>
                <li>मुदत: 15 वर्षे (5 वर्षांच्या ब्लॉकमध्ये विस्तारित करता येते)</li>
                <li>वार्षिक ठेव: ₹500 - ₹1.5 लाख</li>
                <li>व्याज दर: ~7.1% (वार्षिक चक्रवाढ, कर-मुक्त)</li>
                <li>कर लाभ: ईईई (गुंतवणूक, व्याज आणि परिपक्वता सर्व 80C अंतर्गत सूट)</li>
                <li>सर्व भारतीय रहिवाशांसाठी उपलब्ध</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # SSY in Marathi
        st.markdown("""
        <div class="scheme-card">
            <h3>4. सुकन्या समृद्धी योजना (SSY)</h3>
            <p><strong>वेबसाइट:</strong> <a href="https://www.indiapost.gov.in" target="_blank">https://www.indiapost.gov.in</a></p>
            <p><strong>तपशील:</strong></p>
            <ul>
                <li>बालिकेसाठी बचत योजना</li>
                <li>पात्रता: 10 वर्षांखालील बालिका + पालक/पालक खातेदार म्हणून</li>
                <li>मुदत: बालिका 21 वर्षांची होईपर्यंत किंवा 18 वर्षांनंतर लग्न</li>
                <li>वार्षिक ठेव: ₹250 - ₹1.5 लाख</li>
                <li>व्याज दर: ~8.2% (वार्षिक चक्रवाढ, कर-मुक्त)</li>
                <li>कर लाभ: 80C कपात</li>
                <li>प्रति बालिका फक्त एक खाते (कुटुंबातील जुळ्या/तिप्पटांसाठी अपवाद वगळता कमाल 2)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional schemes in Marathi
        st.markdown("""
        <div class="scheme-card">
            <h3>5. प्रधानमंत्री सुरक्षा बीमा योजना (PMSBY)</h3>
            <p><strong>वेबसाइट:</strong> <a href="https://jansuraksha.gov.in" target="_blank">https://jansuraksha.gov.in</a></p>
            <p><strong>तपशील:</strong></p>
            <ul>
                <li>अपघाती मृत्यू आणि अपंगत्व विमा योजना</li>
                <li>कव्हरेज: अपघाती मृत्यू/कायमचे एकूण अपंगत्वासाठी ₹2 लाख</li>
                <li>प्रीमियम: ₹20 प्रति वर्ष (बँक खात्यातून स्वयं-डेबिट)</li>
                <li>वयोगट: 18-70 वर्षे</li>
                <li>वार्षिक नूतनीकरण करता येते</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="scheme-card">
            <h3>6. प्रधानमंत्री जीवन ज्योती बीमा योजना (PMJJBY)</h3>
            <p><strong>वेबसाइट:</strong> <a href="https://jansuraksha.gov.in" target="_blank">https://jansuraksha.gov.in</a></p>
            <p><strong>तपशील:</strong></p>
            <ul>
                <li>जीवन विमा योजना</li>
                <li>कव्हरेज: ₹2 लाख जीवन विमा कव्हर</li>
                <li>प्रीमियम: ₹330 प्रति वर्ष (बँक खात्यातून स्वयं-डेबिट)</li>
                <li>वयोगट: 18-50 वर्षे (55 वर्षांपर्यंत नूतनीकरण करता येते)</li>
                <li>कोणत्याही कारणास्तव जोखीम कव्हरेज</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()