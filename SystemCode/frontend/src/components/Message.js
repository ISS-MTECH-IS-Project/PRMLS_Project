import { Box, Grid } from "@mui/material";
import Paper from "@mui/material/Paper";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
const Message = ({ message, showDetails }) => {
  return (
    <Grid container direction="row" p={1}>
      <Grid item display="inline-flex">
        <Grid container direction="column" alignContent="flex-start" p={1}>
          <Grid item alignItems="flex">
            <img
              alt="Spot My Fish"
              src="/static/images/sample/favicon-32x32.png"
            />
            <span>{message.file.name}</span>
          </Grid>
          <Grid item display="inline-flex">
            <Paper>
              <Box>
                <img alt="User Input" width={400} src={message.preview} />
                <p>
                  Classification Output:{" "}
                  <b>{message.response.result[0].type}</b>
                  {"("}
                  {message.response.result[0].probability.toFixed(2)}%{")"}
                </p>
              </Box>
            </Paper>
          </Grid>
        </Grid>
      </Grid>
      <Grid item display="inline-flex">
        {showDetails && (
          <TableContainer component={Paper}>
            <Table aria-label="simple table">
              <TableHead>
                <TableRow>
                  <TableCell>Model Name</TableCell>
                  <TableCell align="left">Classify Output</TableCell>
                  <TableCell align="right">Probability</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {message.response.result.map((row, i) => (
                  <TableRow
                    key={i}
                    sx={{
                      "&:last-child td, &:last-child th": { border: 0 },
                    }}
                  >
                    <TableCell component="th" scope="row">
                      {row.model_name}
                    </TableCell>
                    <TableCell align="left">{row.type}</TableCell>
                    <TableCell align="right">
                      {row.probability.toFixed(2)}%
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </Grid>
    </Grid>
  );
};

export default Message;
