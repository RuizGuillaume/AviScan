from github import Github
import os
from dotenv import load_dotenv

load_dotenv()


def upload_to_github(file_path, repo_name="MaxenceRemy/Bird-Image-Recognition", branch="main"):
    g = Github(os.getenv("GITHUB_TOKEN"))
    repo = g.get_repo(repo_name)

    with open(file_path, "rb") as file:
        content = file.read()

    file_name = os.path.basename(file_path)
    github_file_path = f"unknown/{file_name}"

    try:
        contents = repo.get_contents(github_file_path, ref=branch)
        repo.update_file(
            github_file_path,
            f"Update {file_name}",
            content,
            contents.sha,
            branch=branch,
        )
    except Exception:
        repo.create_file(github_file_path, f"Add {file_name}", content, branch=branch)

    return f"https://github.com/{repo.full_name}/blob/{branch}/{github_file_path}"
