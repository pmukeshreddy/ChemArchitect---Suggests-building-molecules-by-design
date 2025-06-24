def normalise_applications(df,app_column="applications"):
    norm_col = f"{app_column}_normalised"
    df[norm_col] = df[app_column].astype(str).str.lower()
    
    df[norm_col] = df[norm_col].str.replace('_', ' ', regex=False)
    df[norm_col] = df[norm_col].str.replace('-', ' ', regex=False)

    df[norm_col] = df[norm_col].apply(lambda x: re.sub(r'[^a-z0-9 ]', '', x))
    df[norm_col] = df[norm_col].apply(lambda x: ' '.join(x.split()))
    # This line correctly replaces the specific string.
    df.loc[df[norm_col] == 's disease treatment', norm_col] = 'disease treatment'


    df[norm_col] = df[norm_col].fillna('')
    return df
