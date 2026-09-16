import pandas as pd
from io import StringIO

# Preparing the Data

def load_monthly_block(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the header row: first line starting with a comma
    start_idx = next(i for i, line in enumerate(lines) if line.startswith(','))

    # Find where this block ends: first blank line after the header
    end_idx = next(
        i for i in range(start_idx + 1, len(lines))
        if lines[i].strip() == ''
    )

    block = ''.join(lines[start_idx:end_idx])
    df = pd.read_csv(StringIO(block))

    df = df.rename(columns={df.columns[0]: 'date'})
    df.columns = [c.strip() for c in df.columns]
    df['date'] = df['date'].astype(str).str.strip()

    # Keep only monthly rows (YYYYMM, 6 digits) — drops any stray annual rows
    df = df[df['date'].str.match(r'^\d{6}$')]
    df['date'] = df['date'].astype(int)
    return df.set_index('date')


# Load each file
portfolios = load_monthly_block('Data/portfolios_25_size_mom.csv')
momentum = load_monthly_block('Data/factor_momentum.csv')[['Mom']]
ff3 = load_monthly_block('Data/factor_ff3.csv')[['Mkt-RF', 'SMB', 'RF']]

# Merge on date
data = portfolios.join(ff3, how='inner').join(momentum, how='inner')

# Trim to your sample period
data = data.loc[197208:202409]

print("Shape:", data.shape)
print(data.head())
print(data.tail())

# Compute excess returns for the 25 portfolios
portfolio_cols = portfolios.columns.tolist()
excess = data[portfolio_cols].sub(data['RF'], axis=0)
excess.columns = [f"{c}_excess" for c in excess.columns]

data = pd.concat([data, excess], axis=1)

# Save the cleaned, merged dataset for later use
data.to_csv('Data/merged_clean.csv')
print("\nSaved merged dataset to Data/merged_clean.csv")