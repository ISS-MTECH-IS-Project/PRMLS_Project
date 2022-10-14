import SmartToyIcon from "@mui/icons-material/SmartToy";
import { Box } from "@mui/material";
import Grid from "@mui/material/Grid";

const Topbar = () => {
  return (
    <Grid container direction="row" alignItems="center">
      <img height={25} src="/static/images/sample/favicon-32x32.png" />
      <Box m={2} pt={3}></Box>
      Spot My Fish is here to serve
    </Grid>
  );
};
export default Topbar;
