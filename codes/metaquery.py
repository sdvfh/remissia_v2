from remissia.utils import columns, get_path, get_spark

path = get_path()
spark = get_spark()

file_csv = list(path["datasets"].rglob("*.csv"))[0]
spark.read.csv(str(file_csv), sep=",", header=True).createOrReplaceTempView("SEER")

main_query = "SELECT \n"
for i, column in enumerate(columns):
    if column == "Patient ID":
        continue
    print(column, f"{i + 1}/{len(columns)}")
    main_query += f"-- {column}\n"
    name = f"`{column}`"
    # 8721474
    query = f"""
    SELECT {name}, count(*) as QTD, count(*) / 100000 as PERC
    FROM SEER
    GROUP BY {name}
    order by count(*) desc
    """

    df = spark.sql(query).toPandas()

    df["name"] = (
        df[column]
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.strip()
        .str.replace("/", "")
        .str.replace(")", "")
        .str.replace("(", "")
        .str.replace(".", "")
        .str.replace(";", "")
        .str.replace(":", "")
        .str.replace(",", "")
        .str.replace("$", "")
        .str.replace("+", "E")
        .str.replace("-", "")
        .str.replace("~", "")
        .str.replace(" ", "_")
        .str.replace("__", "_")
        .str.upper()
    )

    for _, line in df.iterrows():
        value = line[column]
        qtd = line["QTD"]
        perc = line["PERC"]
        name_column = line["name"]
        main_query += f"""       CASE WHEN {name} = '{value}' THEN 1 ELSE 0 END AS {name_column}, "
        "--  QTD: {qtd}, {round(perc * 100, 2)}%\n"""

main_query = main_query[:-2] + "\nFROM SEER"

with open(path["queries"] / "all_attributes.sql", "w") as file:
    file.write(main_query)
