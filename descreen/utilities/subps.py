import subprocess as sp


def halftonecv(input_img: bytes, args: list[str]) -> bytes:
    try:
        cp = sp.run(
            ["halftonecv", "-", "-O", "-q"] + args,
            check=True,
            text=False,
            capture_output=True,
            input=input_img,
        )
        return cp.stdout
    except sp.CalledProcessError as e:
        e.returncode
        match e.stderr:
            case str() as stderr:
                pass
            case bytes() as bstderr:
                bstderr.decode()
        raise


def magickpng(input_img: bytes, args: list[str], *, png48: bool = False) -> bytes:
    try:
        cp = sp.run(
            ["magick", "-"] + args + ["PNG48:-" if png48 else "PNG24:-"],
            check=True,
            text=False,
            capture_output=True,
            input=input_img,
        )
        return cp.stdout
    except sp.CalledProcessError as e:
        e.returncode
        match e.stderr:
            case str() as stderr:
                pass
            case bytes() as bstderr:
                bstderr.decode()
        raise
