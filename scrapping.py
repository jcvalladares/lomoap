# Website crawler for RAG data preparation
import os
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import re
from urllib.robotparser import RobotFileParser

from urllib.parse import urlunparse, parse_qsl, urlencode
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def normalize_url(url):
    """Normalize URL by removing fragments and sorting query parameters."""
    parsed = urlparse(url)
    # Remove fragment
    parsed = parsed._replace(fragment="")
    # Sort query params
    query = urlencode(sorted(parse_qsl(parsed.query)))
    parsed = parsed._replace(query=query)
    return urlunparse(parsed)

# robots.txt cache: origin -> (RobotFileParser, crawl_delay_seconds)
_ROBOTS_CACHE = {}


def get_robots_for_origin(origin_url, user_agent="MultiModelScraper/1.0"):
    """Return a (RobotFileParser, crawl_delay_seconds) tuple for the origin.

    This function caches results per origin. It uses `requests` to fetch
    the raw robots.txt so we can also extract a `Crawl-delay` directive
    (RobotFileParser does not expose crawl-delay).
    """
    parsed = urlparse(origin_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if origin in _ROBOTS_CACHE:
        return _ROBOTS_CACHE[origin]

    robots_url = urljoin(origin, "/robots.txt")
    rp = RobotFileParser()
    crawl_delay = None
    try:
        resp = requests.get(robots_url, headers={"User-Agent": user_agent}, timeout=5)
        if resp.status_code == 200:
            # Feed parser and also try to extract Crawl-delay
            rp.parse(resp.text.splitlines())
            # Simple Crawl-delay extraction: look for the first matching user-agent block
            lines = resp.text.splitlines()
            current_ua = None
            candidate_cd = None
            for raw in lines:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("user-agent:"):
                    current_ua = line.split(":", 1)[1].strip()
                    continue
                if line.lower().startswith("crawl-delay:"):
                    try:
                        val = float(line.split(":", 1)[1].strip())
                    except Exception:
                        continue
                    # Prefer explicit match, otherwise accept generic
                    if current_ua and (current_ua == "*" or current_ua.lower() in user_agent.lower()):
                        crawl_delay = val
                        break
                    if candidate_cd is None:
                        candidate_cd = val
            if crawl_delay is None:
                crawl_delay = candidate_cd
        else:
            # fallback to RobotFileParser's reading behavior
            rp.set_url(robots_url)
            rp.read()
    except Exception:
        try:
            rp.set_url(robots_url)
            rp.read()
        except Exception:
            # give up; leave defaults
            pass

    _ROBOTS_CACHE[origin] = (rp, crawl_delay)
    return rp, crawl_delay

# robots.txt cache: origin -> (RobotFileParser, crawl_delay_seconds)

def is_valid_url(url, base_netloc):
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and parsed.netloc == base_netloc

def save_html(url, html, output_dir):
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if not path or path.endswith("/"):
        path += "index.html"
    if not path.endswith(".html"):
        path += ".html"
    save_path = os.path.join(output_dir, path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)

def crawl_website(start_url, output_dir="downloaded_site", max_pages=100):
    visited = set()
    to_visit = [normalize_url(start_url)]
    base_netloc = urlparse(start_url).netloc
    count = 0
    # Respect robots.txt for the origin
    user_agent = "MultiModelScraper/1.0"
    rp, crawl_delay = get_robots_for_origin(start_url, user_agent=user_agent)
    headers = {"User-Agent": user_agent}
    while to_visit and count < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        # Respect robots.txt allow/disallow
        try:
            if not rp.can_fetch(user_agent, url):
                print(f"Skipping (disallowed by robots.txt): {url}")
                continue
        except Exception:
            # if robots parser fails, continue cautiously
            pass
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                continue
            html = resp.text
            save_html(url, html, output_dir)
            visited.add(url)
            count += 1
            soup = BeautifulSoup(html, "html.parser")
            for link in soup.find_all("a", href=True):
                abs_url = urljoin(url, link["href"])
                abs_url = normalize_url(abs_url)
                if is_valid_url(abs_url, base_netloc) and abs_url not in visited and abs_url not in to_visit:
                    # check robots for this URL
                    try:
                        if not rp.can_fetch(user_agent, abs_url):
                            print(f"Skipping link (disallowed by robots.txt): {abs_url}")
                            continue
                    except Exception:
                        pass
                    to_visit.append(abs_url)
                    # we don't pre-fetch the linked page here; it will be visited in the normal loop
            print(f"Crawled: {url}")
            # respect crawl-delay if specified
            if crawl_delay:
                time.sleep(crawl_delay)
        except Exception as e:
            print(f"Failed to crawl {url}: {e}")
    print(f"Crawling complete. {count} pages downloaded.")

def extract_text_from_html_folder(html_folder, output_file="all_text.txt"):
    """Extracts visible text from all HTML files in a folder and all subfolders, writes to a single file."""
    from bs4 import BeautifulSoup
    text_chunks = []
    file_count = 0
    for root, _, files in os.walk(html_folder):
        for file in files:
            if file.endswith(".html"):
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")  # Debug print
                file_count += 1
                with open(file_path, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f, "html.parser")
                    # Remove script and style elements
                    for tag in soup(["script", "style", "noscript"]):
                        tag.decompose()
                    text = soup.get_text(separator=" ", strip=True)
                    if text:
                        text_chunks.append(f"--- {file_path} ---\n{text}\n")
    print(f"Total HTML files processed: {file_count}")
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("\n".join(text_chunks))
    print(f"Extracted text from HTML files in {html_folder} (including subfolders) into {output_file}")

# Example function to use Selenium to enter a zip code and get HTML
def get_html_with_zip(url, zip_code, zip_input_selector, submit_selector=None, wait_time=3):
    """
    Opens the page with Selenium, enters the zip code, submits the form, and returns the resulting HTML.
    zip_input_selector: CSS selector for the zip code input field
    submit_selector: CSS selector for the submit button (optional, if pressing Enter is not enough)
    wait_time: seconds to wait for the page to load after submitting
    """
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    try:
        driver.get(url)
        time.sleep(2)  # Wait for page to load
        zip_input = driver.find_element(By.CSS_SELECTOR, zip_input_selector)
        zip_input.clear()
        zip_input.send_keys(zip_code)
        if submit_selector:
            submit_btn = driver.find_element(By.CSS_SELECTOR, submit_selector)
            submit_btn.click()
        else:
            zip_input.send_keys(Keys.RETURN)
        time.sleep(wait_time)  # Wait for new content to load
        html = driver.page_source
        return html
    finally:
        driver.quit()

def click_all_nav_links(driver, save_html_callback=None, output_dir="downloaded_site"):
    """
    Clicks all top-level nav links and their dropdown items. Optionally saves HTML after each click.
    Handles StaleElementReferenceException by re-finding elements after navigation.
    """
    import time
    from selenium.common.exceptions import StaleElementReferenceException
    from selenium.webdriver.common.action_chains import ActionChains

    actions = ActionChains(driver)
    nav_selector = "nav.navbar"
    link_selector = "ul.navbar-nav > li > a.nav-link"
    dropdown_selector = "ul.dropdown-menu a.dropdown-item"

    nav = driver.find_element(By.CSS_SELECTOR, nav_selector)
    nav_links = nav.find_elements(By.CSS_SELECTOR, link_selector)
    num_links = len(nav_links)
    i = 0
    while i < num_links:
        try:
            nav = driver.find_element(By.CSS_SELECTOR, nav_selector)
            nav_links = nav.find_elements(By.CSS_SELECTOR, link_selector)
            link = nav_links[i]
            title = link.get_attribute("title")
            href = link.get_attribute("href")
            print(f"Clicking nav: {title} ({href})")
            actions.move_to_element(link).perform()
            link.click()
            time.sleep(2)
            if save_html_callback:
                save_html_callback(driver.current_url, driver.page_source, output_dir)
            # Handle dropdowns
            dropdown_items = driver.find_elements(By.CSS_SELECTOR, dropdown_selector)
            for j, item in enumerate(dropdown_items):
                try:
                    dropdown_items = driver.find_elements(By.CSS_SELECTOR, dropdown_selector)
                    item = dropdown_items[j]
                    item_title = item.get_attribute("title")
                    item_href = item.get_attribute("href")
                    print(f"  Clicking dropdown: {item_title} ({item_href})")
                    driver.execute_script("arguments[0].scrollIntoView();", item)
                    item.click()
                    time.sleep(2)
                    if save_html_callback:
                        save_html_callback(driver.current_url, driver.page_source, output_dir)
                    driver.back()
                    time.sleep(2)
                    nav = driver.find_element(By.CSS_SELECTOR, nav_selector)
                    nav_links = nav.find_elements(By.CSS_SELECTOR, link_selector)
                    link = nav_links[i]
                    actions.move_to_element(link).perform()
                except StaleElementReferenceException:
                    print("Stale dropdown item, retrying...")
                    time.sleep(1)
                    continue
            driver.back()
            time.sleep(2)
            i += 1
        except StaleElementReferenceException:
            print("Stale nav element, retrying...")
            time.sleep(1)
            continue

def handle_zip_modal(driver, zip_code="33166", wait_time=2):
    """
    Checks for the zip code modal, enters the zip code, and clicks the button if visible.
    """
    import time
    try:
        modal = driver.find_element(By.CSS_SELECTOR, "div#zip-location.show")
        if modal.is_displayed():
            print("Zip modal is visible.")
            zip_input = modal.find_element(By.CSS_SELECTOR, "input.location_zipcode")
            zip_input.clear()
            zip_input.send_keys(zip_code)
            time.sleep(0.5)
            button = modal.find_element(By.CSS_SELECTOR, "button.loct-zipcode-btn")
            button.click()
            print(f"Entered zip code {zip_code} and clicked the button.")
            time.sleep(wait_time)
        else:
            print("Zip modal is not visible.")
    except NoSuchElementException:
        print("Zip modal not found.")

def crawl_with_modal_and_nav(
    start_url="https://www.delair.com/",
    output_dir="downloaded_site",
    zip_code="33166",
    max_nav_links=20
):
    """
    Loads the page, handles the zip modal if present, clicks all nav links, and saves HTML after each.
    """
   
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    import time

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    try:
        driver.get(start_url)
        time.sleep(2)
        handle_zip_modal(driver, zip_code=zip_code, wait_time=2)
        click_all_nav_links(driver, save_html_callback=save_html, output_dir=output_dir)
    finally:
        driver.quit()
    print(f"Crawling with modal and nav complete. HTML files saved in {output_dir}")

def crawl_main_and_nav_with_modal(driver, output_dir="downloaded_site", zip_code="33166"):
    """
    Save the main page, then click each nav link (anchor inside li) using JS click, handle modal if it appears, and save the resulting page.
    """
    import time
    from selenium.common.exceptions import ElementClickInterceptedException, StaleElementReferenceException

    # 1. Save the main page
    save_html(driver.current_url, driver.page_source, output_dir)

    # 2. Click each nav link (anchor inside li), handle modal if it appears, and save the resulting page
    nav_selector = "nav.navbar"
    li_selector = "ul.navbar-nav > li"
    anchor_selector = "a.nav-link"
    nav = driver.find_element(By.CSS_SELECTOR, nav_selector)
    nav_lis = nav.find_elements(By.CSS_SELECTOR, li_selector)
    num_links = len(nav_lis)
    for i in range(num_links):
        try:
            nav = driver.find_element(By.CSS_SELECTOR, nav_selector)
            nav_lis = nav.find_elements(By.CSS_SELECTOR, li_selector)
            li = nav_lis[i]
            anchor = li.find_element(By.CSS_SELECTOR, anchor_selector)
            title = anchor.get_attribute("title")
            href = anchor.get_attribute("href")
            print(f"Clicking nav: {title} ({href})")
            driver.execute_script("arguments[0].scrollIntoView();", anchor)
            time.sleep(0.5)
            try:
                driver.execute_script("arguments[0].click();", anchor)
            except ElementClickInterceptedException:
                print("Modal appeared, handling zip modal...")
                handle_zip_modal(driver, zip_code=zip_code)
                time.sleep(1)
                anchor = driver.find_elements(By.CSS_SELECTOR, li_selector)[i].find_element(By.CSS_SELECTOR, anchor_selector)
                driver.execute_script("arguments[0].click();", anchor)
            time.sleep(2)
            # After click, check for modal again (in case it appears after navigation)
            handle_zip_modal(driver, zip_code=zip_code)
            time.sleep(1)
            save_html(driver.current_url, driver.page_source, output_dir)
            driver.back()
            time.sleep(2)
        except StaleElementReferenceException:
            print("Stale nav element, retrying...")
            time.sleep(1)
            continue

def crawl_and_extract_all_text(
    driver,
    output_dir="downloaded_site",
    zip_code="33166",
    text_output_file="all_text.txt"
):
    """
    1. Crawl all pages without modal and extract text.
    2. Click each nav item, enter zip code if modal appears, extract text from resulting page.
    3. Save all extracted text in one file.
    """
    import time
    from selenium.common.exceptions import ElementClickInterceptedException, StaleElementReferenceException, TimeoutException
    from bs4 import BeautifulSoup
    all_text_chunks = []
    visited_urls = set()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save and extract text from the main page (no modal)
    main_url = driver.current_url
    main_html = driver.page_source
    soup = BeautifulSoup(main_html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    main_text = soup.get_text(separator=" ", strip=True)
    if main_text:
        all_text_chunks.append(f"--- {main_url} ---\n{main_text}\n")
    visited_urls.add(main_url)
    # Optionally save HTML
    with open(os.path.join(output_dir, "main.html"), "w", encoding="utf-8") as f:
        f.write(main_html)

    # 2. Click each nav link (anchor inside li), handle modal if it appears, extract text
    nav_selector = "nav.navbar"
    li_selector = "ul.navbar-nav > li"
    anchor_selector = "a.nav-link"
    nav = driver.find_element(By.CSS_SELECTOR, nav_selector)
    nav_lis = nav.find_elements(By.CSS_SELECTOR, li_selector)
    num_links = len(nav_lis)
    for i in range(num_links):
        try:
            nav = driver.find_element(By.CSS_SELECTOR, nav_selector)
            nav_lis = nav.find_elements(By.CSS_SELECTOR, li_selector)
            li = nav_lis[i]
            anchor = li.find_element(By.CSS_SELECTOR, anchor_selector)
            title = anchor.get_attribute("title")
            href = anchor.get_attribute("href")
            print(f"Clicking nav: {title} ({href})")
            driver.execute_script("arguments[0].scrollIntoView();", anchor)
            time.sleep(0.5)
            prev_url = driver.current_url
            try:
                driver.execute_script("arguments[0].click();", anchor)
            except ElementClickInterceptedException:
                print("Modal appeared, handling zip modal...")
                handle_zip_modal(driver, zip_code=zip_code)
                time.sleep(1)
                anchor = driver.find_elements(By.CSS_SELECTOR, li_selector)[i].find_element(By.CSS_SELECTOR, anchor_selector)
                driver.execute_script("arguments[0].click();", anchor)
            # Wait for URL to change or timeout
            try:
                WebDriverWait(driver, 10).until(lambda d: d.current_url != prev_url)
            except TimeoutException:
                print("Timeout waiting for navigation. Skipping.")
                continue
            time.sleep(2)
            # After click, check for modal again (in case it appears after navigation)
            handle_zip_modal(driver, zip_code=zip_code)
            time.sleep(1)
            new_url = driver.current_url
            if new_url not in visited_urls:
                html = driver.page_source
                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                page_text = soup.get_text(separator=" ", strip=True)
                if page_text:
                    all_text_chunks.append(f"--- {new_url} ---\n{page_text}\n")
                visited_urls.add(new_url)
                # Save HTML
                safe_name = title.replace(" ", "_").replace("/", "_")
                with open(os.path.join(output_dir, f"{safe_name}.html"), "w", encoding="utf-8") as f:
                    f.write(html)
            else:
                print(f"Already visited {new_url}, skipping text extraction.")
            driver.back()
            # Wait for URL to return to main
            try:
                WebDriverWait(driver, 10).until(lambda d: d.current_url == main_url)
            except TimeoutException:
                print("Timeout waiting to return to main page.")
            time.sleep(2)
        except StaleElementReferenceException:
            print("Stale nav element, retrying...")
            time.sleep(1)
            continue

    # 3. Save all text to one file
    with open(text_output_file, "w", encoding="utf-8") as out:
        out.write("\n".join(all_text_chunks))
    print(f"Extracted text from all pages into {text_output_file}")

# Example main workflow
if __name__ == "__main__":
    extract_text_from_html_folder(html_folder="downloaded_site", output_file="all_text.txt")


    # crawl_website("https://www.delair.com/", output_dir="downloaded_site", max_pages=150)
    # from selenium import webdriver
    # from selenium.webdriver.chrome.service import Service
    # from webdriver_manager.chrome import ChromeDriverManager
    # import time
    # service = Service(ChromeDriverManager().install())
    # driver = webdriver.Chrome(service=service)
    # try:
    #     driver.get("https://www.delair.com/")
    #     time.sleep(2)
       
    #    # crawl_and_extract_all_text(driver, output_dir="downloaded_site", zip_code="33166", text_output_file="all_text.txt")
    # finally:
    #     driver.quit()
    # print("Crawling and text extraction complete. All text saved in all_text.txt")
