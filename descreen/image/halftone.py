import halftonecv.cli as hcv


def convert(input_img: bytes, args: list[str]) -> bytes:
    result: bytes | None = None

    def receive(img_bytes: bytes) -> None:
        nonlocal result
        if result is None:
            result = img_bytes
        else:
            raise RuntimeError()

    try:
        code: int = hcv.main(argv=(["-q", "-V"] + args), inputs=[input_img], refout=receive, nofile=True, notrap=True)
    except SystemExit as e:
        raise RuntimeError() from e
    if code != 0:
        raise RuntimeError(code)
    if result is None:
        raise RuntimeError()
    return result
