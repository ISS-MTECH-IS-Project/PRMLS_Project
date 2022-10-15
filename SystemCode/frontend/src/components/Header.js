import Grid from "@mui/material/Grid";
const Header = () => {
  return (
    <Grid container direction="row" alignItems="center" p={1}>
      <Grid item xs={11}>
        <h3>Welcome to Spot My Fish</h3>
      </Grid>
      <Grid item xs={1}>
        <img
          height={100}
          alt="Spot My Fish"
          src="/static/images/sample/android-chrome-192x192.png"
        />
      </Grid>
    </Grid>
  );
};

export default Header;
