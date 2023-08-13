MORPHO_DIR="morphodita/"
DWNLD_LINK='https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-4794{/czech-morfflex2.0-pdtc1.0-220710.zip}'

mkdir -p "$MORPHO_DIR" && cd "$MORPHO_DIR"
curl --remote-name-all "$DWNLD_LINK"
unzip *.zip
rm *.zip
