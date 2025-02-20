from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd
from playwright._impl._errors import TimeoutError

def scrape_table_data(page):
    html = page.inner_html("body")
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("tbody.Crom_body__UYOcU tr")
    data = [[td.get_text() for td in row.find_all("td")] for row in rows]
    return data

def scrape_season_data(page, season):
    url = f"https://www.nba.com/stats/players/boxscores?Season={season}"
    page.goto(url, timeout=120000)

    
    try:
        page.wait_for_selector("tbody.Crom_body__UYOcU tr", timeout=30000)
    except TimeoutError:
        print(f"Page did not load properly for season {season}. Skipping...")
        return []

    data = []
    page_number = 1

    while True:
        try:
            data += scrape_table_data(page)
            print(f"Scraped data from page {page_number} of season {season}.")

            next_button = page.query_selector("button[data-pos='next']")
            if not next_button or not next_button.is_enabled():
                print(f"No next button found for season {season}. Exiting loop.")
                break

            next_button.click()
            page.wait_for_selector("tbody.Crom_body__UYOcU tr", timeout=15000)
            page_number += 1

        except TimeoutError:
            print(f"Timeout error while navigating pages in season {season}. Retrying...")
            continue
        except Exception as e:
            print(f"An error occurred while scraping season {season}: {e}")
            break

    return data


def main():
    seasons = ["2024-25", "2023-24", "2022-23", "2021-22", "2020-21", "2019-20"]
    
    columns = [
        "Player", "Team", "Opponent", "Date", "Result", "Minutes", "Points", 
        "FGM", "FGA", "FG%", "3PM", "3PA", "3P%", "FTM", "FTA", "FT%", 
        "OREB", "DREB", "REB", "AST", "TO", "STL", "BLK", "PF", "+/-", "SPI"
    ]
    
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        all_data = []

        for season in seasons:
            print(f"Starting to scrape data for the {season} season...")
            season_data = scrape_season_data(page, season)
            all_data += season_data

        df = pd.DataFrame(all_data, columns=columns) 

        df.to_csv("nba_player_boxscores_multiple_seasons.csv", index=False)
        print("Scraping completed for all seasons.")


main()
