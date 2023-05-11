import requests
from bs4 import BeautifulSoup


def main() -> int:
    url = "https://leagueoflegends.fandom.com/wiki/List_of_champions"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    img_tags = soup.find_all("img", alt=lambda x: x and "OriginalSquare.png" in x)
    filtered_img_tags = [img["data-src"].rsplit(".png", 1)[0] + ".png" for img in img_tags]

    for img_url in filtered_img_tags:
        champion_name = img_url.split("/")[-1].split("_OriginalSquare")[0]
        response = requests.get(img_url)
        with open(f"data/{champion_name}.png", "wb") as f:
            f.write(response.content)

    return 0


if __name__ == "__main__":
    main()
