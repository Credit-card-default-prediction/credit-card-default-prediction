import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Professional fintech palette
COLORS = {
    'primary': '#1E3A8A',    # Navy
    'secondary': '#3B82F6',  # Bright Blue
    'danger': '#EF4444',     # Crimson / Default
    'success': '#10B981',    # Emerald / Non-default
    'warning': '#F59E0B',    # Amber
    'neutral': '#64748B',    # Slate
    'dark': '#0F172A',       # Slate Dark
    'light': '#F8FAFC'       # Slate Light
}

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 0.8


def get_summary_stats(df):
    """Return comprehensive summary statistics for the dataset."""
    numeric_df = df.select_dtypes(include=[np.number])
    stats = numeric_df.describe().T
    stats['missing_values'] = df[numeric_df.columns].isnull().sum()
    stats['missing_pct'] = (stats['missing_values'] / len(df)) * 100
    stats['skewness'] = numeric_df.skew()
    return stats.round(2)


def plot_default_distribution(df):
    """Plot target distribution (Default vs Non-Default) with counts & percentages."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    counts = df['default'].value_counts()
    labels = ['Non-Default (Will Pay)', 'Default (At Risk)']
    colors = [COLORS['success'], COLORS['danger']]
    
    # Donut Chart
    wedges, texts, autotexts = axes[0].pie(
        counts, labels=labels, autopct='%1.1f%%',
        startangle=140, colors=colors,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
        textprops=dict(color='#1E293B', fontsize=11, fontweight='bold')
    )
    axes[0].set_title('Target Class Proportion (30,000 Clients)', fontsize=13, fontweight='bold', pad=15)
    
    # Bar Chart
    bars = axes[1].bar(labels, counts.values, color=colors, width=0.5, edgecolor='#1E293B', linewidth=1)
    axes[1].set_title('Client Count by Default Status', fontsize=13, fontweight='bold', pad=15)
    axes[1].set_ylabel('Number of Clients', fontsize=11)
    for bar in bars:
        h = bar.get_height()
        axes[1].annotate(f'{h:,} ({h/len(df)*100:.1f}%)',
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 5), textcoords="offset points",
                         ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_demographics_impact(df):
    """Plot default rates across demographic features: Gender, Education, and Marriage."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. Gender (1=Male, 2=Female)
    gender_map = {1: 'Male', 2: 'Female'}
    df_temp = df.copy()
    df_temp['SEX_LABEL'] = df_temp['SEX'].map(gender_map)
    gender_rate = df_temp.groupby('SEX_LABEL')['default'].agg(['count', 'mean']).reset_index()
    gender_rate['mean'] *= 100
    
    bars1 = axes[0].bar(gender_rate['SEX_LABEL'], gender_rate['mean'], color=[COLORS['secondary'], '#EC4899'], width=0.45)
    axes[0].set_title('Default Rate by Gender', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Default Rate (%)', fontsize=10)
    axes[0].set_ylim(0, 30)
    for bar in bars1:
        axes[0].annotate(f'{bar.get_height():.1f}%',
                         xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                         xytext=(0, 4), textcoords="offset points", ha='center', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # 2. Education (1=Grad School, 2=University, 3=High School, 4=Others)
    edu_map = {1: 'Grad School', 2: 'University', 3: 'High School', 4: 'Others', 5: 'Others', 6: 'Others', 0: 'Others'}
    df_temp['EDU_LABEL'] = df_temp['EDUCATION'].map(edu_map)
    edu_order = ['Grad School', 'University', 'High School', 'Others']
    edu_rate = df_temp.groupby('EDU_LABEL')['default'].agg(['count', 'mean']).reindex(edu_order).reset_index()
    edu_rate['mean'] *= 100
    
    bars2 = axes[1].bar(edu_rate['EDU_LABEL'], edu_rate['mean'], color='#6366F1', width=0.55)
    axes[1].set_title('Default Rate by Education Level', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Default Rate (%)', fontsize=10)
    axes[1].set_ylim(0, 35)
    axes[1].tick_params(axis='x', rotation=15)
    for bar in bars2:
        axes[1].annotate(f'{bar.get_height():.1f}%',
                         xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                         xytext=(0, 4), textcoords="offset points", ha='center', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # 3. Marriage (1=Married, 2=Single, 3=Others)
    mar_map = {1: 'Married', 2: 'Single', 3: 'Others', 0: 'Others'}
    df_temp['MAR_LABEL'] = df_temp['MARRIAGE'].map(mar_map)
    mar_order = ['Married', 'Single', 'Others']
    mar_rate = df_temp.groupby('MAR_LABEL')['default'].agg(['count', 'mean']).reindex(mar_order).reset_index()
    mar_rate['mean'] *= 100
    
    bars3 = axes[2].bar(mar_rate['MAR_LABEL'], mar_rate['mean'], color='#F59E0B', width=0.45)
    axes[2].set_title('Default Rate by Marital Status', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Default Rate (%)', fontsize=10)
    axes[2].set_ylim(0, 30)
    for bar in bars3:
        axes[2].annotate(f'{bar.get_height():.1f}%',
                         xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                         xytext=(0, 4), textcoords="offset points", ha='center', fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_age_distribution(df):
    """Plot customer age distribution and default rate by age cohort."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distribution KDE
    sns.kdeplot(df[df['default'] == 0]['AGE'], label='Non-Default', color=COLORS['success'], fill=True, alpha=0.35, ax=axes[0])
    sns.kdeplot(df[df['default'] == 1]['AGE'], label='Default', color=COLORS['danger'], fill=True, alpha=0.35, ax=axes[0])
    axes[0].set_title('Age Density by Default Status', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Age (Years)', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].legend(frameon=True)
    axes[0].grid(True, alpha=0.3)
    
    # Age Cohort Bins
    df_temp = df.copy()
    bins = [20, 25, 35, 45, 55, 75]
    labels = ['21-25', '26-35', '36-45', '46-55', '56+']
    df_temp['AGE_GROUP'] = pd.cut(df_temp['AGE'], bins=bins, labels=labels, right=True)
    age_rate = df_temp.groupby('AGE_GROUP', observed=False)['default'].agg(['count', 'mean']).reset_index()
    age_rate['mean'] *= 100
    
    bars = axes[1].bar(age_rate['AGE_GROUP'].astype(str), age_rate['mean'], color=COLORS['primary'], width=0.5)
    axes[1].set_title('Default Probability by Age Group', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Age Bracket', fontsize=11)
    axes[1].set_ylabel('Default Rate (%)', fontsize=11)
    axes[1].set_ylim(0, 35)
    for bar in bars:
        axes[1].annotate(f'{bar.get_height():.1f}%',
                         xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                         xytext=(0, 4), textcoords="offset points", ha='center', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_credit_limit_distribution(df):
    """Plot credit limit (LIMIT_BAL) vs default risk."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    data_to_plot = [df[df['default'] == 0]['LIMIT_BAL'], df[df['default'] == 1]['LIMIT_BAL']]
    box = axes[0].boxplot(data_to_plot, patch_artist=True, tick_labels=['Non-Default', 'Default'],
                          boxprops=dict(linewidth=1.2), medianprops=dict(color='black', linewidth=1.5))
    box['boxes'][0].set_facecolor(COLORS['success'])
    box['boxes'][1].set_facecolor(COLORS['danger'])
    axes[0].set_title('Credit Limit (NT$) Distribution', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Credit Limit (NT$)', fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Credit Limit Deciles Default Rate
    df_temp = df.copy()
    df_temp['LIMIT_TIER'] = pd.qcut(df_temp['LIMIT_BAL'], q=5, labels=['Tier 1 (Lowest)', 'Tier 2', 'Tier 3', 'Tier 4', 'Tier 5 (Highest)'])
    limit_rate = df_temp.groupby('LIMIT_TIER', observed=False)['default'].mean() * 100
    
    bars = axes[1].bar(limit_rate.index.astype(str), limit_rate.values, color='#0284C7', width=0.55)
    axes[1].set_title('Default Rate by Credit Limit Quintile', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Default Rate (%)', fontsize=11)
    axes[1].set_ylim(0, 40)
    axes[1].tick_params(axis='x', rotation=15)
    for bar in bars:
        axes[1].annotate(f'{bar.get_height():.1f}%',
                         xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                         xytext=(0, 4), textcoords="offset points", ha='center', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_repayment_status_heatmap(df):
    """Heatmap of default rates across repayment status months (PAY_0 through PAY_6)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    month_names = ['Sept 2005 (PAY_0)', 'Aug 2005 (PAY_2)', 'July 2005 (PAY_3)',
                   'June 2005 (PAY_4)', 'May 2005 (PAY_5)', 'April 2005 (PAY_6)']
    
    status_values = [-2, -1, 0, 1, 2, 3, 4]
    heatmap_data = []
    
    for val in status_values:
        row = []
        for col in pay_cols:
            subset = df[df[col] == val]
            rate = subset['default'].mean() * 100 if len(subset) > 10 else np.nan
            row.append(rate)
        heatmap_data.append(row)
        
    status_labels = [
        '-2: No Consumption',
        '-1: Paid In Full',
        ' 0: Revolving / Min Paid',
        ' 1: 1 Month Delay',
        ' 2: 2 Months Delay',
        ' 3: 3 Months Delay',
        ' 4+: 4+ Months Delay'
    ]
    
    heatmap_df = pd.DataFrame(heatmap_data, index=status_labels, columns=month_names)
    sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Default Rate (%)'},
                linewidths=0.8, ax=ax)
    ax.set_title('Delinquency Escalation Matrix: Default Rate (%) by Repayment Status', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Repayment Status Code', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_bill_vs_pay_trends(df):
    """Plot 6-month trajectory of average bill amount vs payment amount for default vs non-default."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    months = ['Apr (M6)', 'May (M5)', 'Jun (M4)', 'Jul (M3)', 'Aug (M2)', 'Sep (M1)']
    bill_cols = ['BILL_AMT6', 'BILL_AMT5', 'BILL_AMT4', 'BILL_AMT3', 'BILL_AMT2', 'BILL_AMT1']
    pay_cols = ['PAY_AMT6', 'PAY_AMT5', 'PAY_AMT4', 'PAY_AMT3', 'PAY_AMT2', 'PAY_AMT1']
    
    non_def = df[df['default'] == 0]
    def_grp = df[df['default'] == 1]
    
    # 1. Average Bill Amount
    axes[0].plot(months, non_def[bill_cols].mean(), marker='o', linewidth=2.5, color=COLORS['success'], label='Non-Default')
    axes[0].plot(months, def_grp[bill_cols].mean(), marker='s', linewidth=2.5, color=COLORS['danger'], linestyle='--', label='Default')
    axes[0].set_title('Average Monthly Statement Bill (NT$)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Bill Amount (NT$)', fontsize=11)
    axes[0].legend(frameon=True)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Average Payment Amount
    axes[1].plot(months, non_def[pay_cols].mean(), marker='o', linewidth=2.5, color=COLORS['success'], label='Non-Default')
    axes[1].plot(months, def_grp[pay_cols].mean(), marker='s', linewidth=2.5, color=COLORS['danger'], linestyle='--', label='Default')
    axes[1].set_title('Average Monthly Repayment Amount (NT$)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Payment Amount (NT$)', fontsize=11)
    axes[1].legend(frameon=True)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_credit_utilization(df):
    """Plot credit utilization distribution and default rate by utilization tier."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    df_temp = df.copy()
    # Credit utilization in September
    df_temp['UTIL_SEPT'] = (df_temp['BILL_AMT1'] / np.maximum(df_temp['LIMIT_BAL'], 1)) * 100
    df_temp['UTIL_SEPT_CLIP'] = df_temp['UTIL_SEPT'].clip(lower=0, upper=150)
    
    sns.kdeplot(df_temp[df_temp['default'] == 0]['UTIL_SEPT_CLIP'], label='Non-Default',
                color=COLORS['success'], fill=True, alpha=0.35, ax=axes[0])
    sns.kdeplot(df_temp[df_temp['default'] == 1]['UTIL_SEPT_CLIP'], label='Default',
                color=COLORS['danger'], fill=True, alpha=0.35, ax=axes[0])
    axes[0].set_title('Credit Utilization Density (September)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Utilization Rate (%) [Bill / Limit]', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].legend(frameon=True)
    axes[0].grid(True, alpha=0.3)
    
    # Binned Utilization
    util_bins = [-1, 10, 30, 60, 90, 300]
    util_labels = ['<10% (Low)', '10-30% (Healthy)', '30-60% (Moderate)', '60-90% (High)', '>90% (Stretched)']
    df_temp['UTIL_TIER'] = pd.cut(df_temp['UTIL_SEPT'], bins=util_bins, labels=util_labels)
    util_rate = df_temp.groupby('UTIL_TIER', observed=False)['default'].mean() * 100
    
    bars = axes[1].bar(util_rate.index.astype(str), util_rate.values, color='#7C3AED', width=0.55)
    axes[1].set_title('Default Rate by Credit Utilization Tier', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Default Rate (%)', fontsize=11)
    axes[1].set_ylim(0, 45)
    axes[1].tick_params(axis='x', rotation=15)
    for bar in bars:
        axes[1].annotate(f'{bar.get_height():.1f}%',
                         xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                         xytext=(0, 4), textcoords="offset points", ha='center', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(df):
    """Plot correlation heatmap with the target variable `default`."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    cols = ['default', 'LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
            'BILL_AMT1', 'BILL_AMT2', 'PAY_AMT1', 'PAY_AMT2']
    corr = df[cols].corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-0.3, vmax=0.4,
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Pearson Correlation'})
    ax.set_title('Correlation Matrix of Core Financial Indicators & Default', fontsize=13, fontweight='bold', pad=12)
    
    plt.tight_layout()
    return fig


def plot_recent_delinquency_impact(df):
    """Plot the dramatic impact of September repayment status (PAY_0) on default probability."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    status_map = {
        -2: 'No Consumption',
        -1: 'Paid Duly',
        0: 'Revolving',
        1: '1 Mo Delay',
        2: '2 Mo Delay',
        3: '3 Mo Delay',
        4: '4+ Mo Delay'
    }
    df_temp = df.copy()
    df_temp['PAY_0_GROUP'] = df_temp['PAY_0'].apply(lambda x: 4 if x >= 4 else x)
    df_temp['STATUS_LABEL'] = df_temp['PAY_0_GROUP'].map(status_map)
    order = ['No Consumption', 'Paid Duly', 'Revolving', '1 Mo Delay', '2 Mo Delay', '3 Mo Delay', '4+ Mo Delay']
    
    summary = df_temp.groupby('STATUS_LABEL')['default'].agg(['count', 'mean']).reindex(order).reset_index()
    summary['mean'] *= 100
    
    palette = [COLORS['success'] if x < 25 else (COLORS['warning'] if x < 50 else COLORS['danger']) for x in summary['mean']]
    bars = ax.bar(summary['STATUS_LABEL'], summary['mean'], color=palette, width=0.55, edgecolor='#1E293B', linewidth=0.8)
    
    ax.set_title('September Repayment Status (PAY_0) vs Probability of Default', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Default Rate (%)', fontsize=11)
    ax.set_ylim(0, 90)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
    plt.tight_layout()
    return fig


def generate_all_eda_charts(df, output_dir):
    """Generate and save all publication-grade charts to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    charts = {}
    
    # 1. Target distribution
    fig1 = plot_default_distribution(df)
    p1 = os.path.join(output_dir, 'default_distribution.png')
    fig1.savefig(p1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    charts['default_distribution'] = p1
    
    # 2. Demographics impact
    fig2 = plot_demographics_impact(df)
    p2 = os.path.join(output_dir, 'demographics_impact.png')
    fig2.savefig(p2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    charts['demographics_impact'] = p2
    
    # 3. Age distribution
    fig3 = plot_age_distribution(df)
    p3 = os.path.join(output_dir, 'age_distribution.png')
    fig3.savefig(p3, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    charts['age_distribution'] = p3
    
    # 4. Credit limit
    fig4 = plot_credit_limit_distribution(df)
    p4 = os.path.join(output_dir, 'credit_limit_distribution.png')
    fig4.savefig(p4, dpi=300, bbox_inches='tight')
    plt.close(fig4)
    charts['credit_limit_distribution'] = p4
    
    # 5. Repayment heatmap
    fig5 = plot_repayment_status_heatmap(df)
    p5 = os.path.join(output_dir, 'repayment_heatmap.png')
    fig5.savefig(p5, dpi=300, bbox_inches='tight')
    plt.close(fig5)
    charts['repayment_heatmap'] = p5
    
    # 6. Bill vs payment trends
    fig6 = plot_bill_vs_pay_trends(df)
    p6 = os.path.join(output_dir, 'bill_vs_pay_trends.png')
    fig6.savefig(p6, dpi=300, bbox_inches='tight')
    plt.close(fig6)
    charts['bill_vs_pay_trends'] = p6
    
    # 7. Credit utilization
    fig7 = plot_credit_utilization(df)
    p7 = os.path.join(output_dir, 'credit_utilization.png')
    fig7.savefig(p7, dpi=300, bbox_inches='tight')
    plt.close(fig7)
    charts['credit_utilization'] = p7
    
    # 8. Correlation matrix
    fig8 = plot_correlation_matrix(df)
    p8 = os.path.join(output_dir, 'correlation_matrix.png')
    fig8.savefig(p8, dpi=300, bbox_inches='tight')
    plt.close(fig8)
    charts['correlation_matrix'] = p8
    
    # 9. Recent delinquency
    fig9 = plot_recent_delinquency_impact(df)
    p9 = os.path.join(output_dir, 'recent_delinquency.png')
    fig9.savefig(p9, dpi=300, bbox_inches='tight')
    plt.close(fig9)
    charts['recent_delinquency'] = p9
    
    print(f"Generated {len(charts)} EDA charts in {output_dir}")
    return charts


if __name__ == '__main__':
    csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_of_credit_card_clients.csv')
    if os.path.exists(csv_file):
        df_sample = pd.read_csv(csv_file)
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_charts')
        generate_all_eda_charts(df_sample, out)
