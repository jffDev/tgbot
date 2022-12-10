import sys

REQ_VERSION = 3


def main():
    env_version = sys.version_info.major

    if env_version != REQ_VERSION:
        raise TypeError(
            "This project requires Python {}. Found: Python {}".format(
                REQ_VERSION, env_version
            )
        )
    else:
        print("Passes all tests!")


if __name__ == "__main__":
    main()
