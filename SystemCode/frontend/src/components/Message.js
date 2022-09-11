import { BiBot } from "react-icons/bi";
import { Box, Card, CardContent, Grid } from "@mui/material";
import Paper from "@mui/material/Paper";

const Message = ({ message }) => {
  return (
    <>
      <Grid container direction="column" alignContent="flex-start">
        <Grid item alignItems="flex">
          <BiBot />
          <span> ChatBot</span>
          <span> @ {message.time}</span>
          {message.hasOwnProperty("symptoms") && message.symptoms.length === 0 && (
            <Card>
              <CardContent>
                Dear user, I apologise for not understanding your response.
                <br />
                Please try to describe your pet fish's symptoms more.
              </CardContent>
            </Card>
          )}
        </Grid>
        <Grid item display="inline-flex">
          <Paper elevation={1}>
            <Box m={1}>
              <span>{message.body}</span>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </>
  );
};

export default Message;
